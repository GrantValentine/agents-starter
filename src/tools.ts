/**
 * Tool definitions for the AI chat agent
 * Tools can either require human confirmation or execute automatically
 */

import { tool, type ToolSet } from "ai";
import { z } from "zod/v3";
import type { Env, Chat } from "./server";
import { getCurrentAgent } from "agents";
import { scheduleSchema } from "agents/schedule";
import { nelderMead } from "./nelderMead"; // your local Nelder–Mead implementation

/* ---------- Constants ---------- */

// We hardcode the curve date as Nov 19, 2025
const CURVE_DATE = new Date("2025-11-19T00:00:00Z");
const DAYS_IN_YEAR = 365.25;

/* ---------- Bond / date helpers ---------- */

function getFrequencyPerYear(
  securityType: string,
  InterestPaymentFrequency: string | null | undefined,
  interestRate: number
): { frequency: number; isBill: boolean } {
  const typeLower = (securityType || "").toLowerCase();
  const freqLower = (InterestPaymentFrequency || "").toLowerCase();

  // Bills or zero-coupon
  if (
    typeLower.includes("bill") ||
    freqLower.includes("none") ||
    interestRate === 0
  ) {
    return { frequency: 0, isBill: true };
  }

  if (freqLower.includes("semi")) return { frequency: 2, isBill: false };
  if (freqLower.includes("quarter")) return { frequency: 4, isBill: false };
  if (freqLower.includes("annual") || freqLower.includes("year")) {
    return { frequency: 1, isBill: false };
  }

  // Default for notes/bonds if not specified
  if (typeLower.includes("note") || typeLower.includes("bond")) {
    return { frequency: 2, isBill: false };
  }

  // Fallback: treat as semi-annual coupon
  return { frequency: 2, isBill: false };
}

function parseISODate(s: string): Date {
  // Handles "YYYY-MM-DD" or "YYYY-MM-DDT00:00:00"
  return new Date(s);
}

function daysBetween(a: Date, b: Date): number {
  const ms = b.getTime() - a.getTime();
  return ms / (1000 * 60 * 60 * 24);
}

function daysInMonth(year: number, monthZeroBased: number): number {
  // monthZeroBased: 0 = Jan, 1 = Feb, ...
  return new Date(year, monthZeroBased + 1, 0).getDate();
}

// Respect "end-of-month" rule (handles Feb 28 -> Feb 29 on leap years)
function addMonthsEOM(d: Date, months: number): Date {
  const year = d.getFullYear();
  const month = d.getMonth();
  const day = d.getDate();
  const dim = daysInMonth(year, month);
  const isEOM = day === dim;

  const nd = new Date(d);
  nd.setMonth(month + months);

  if (isEOM) {
    const newDim = daysInMonth(nd.getFullYear(), nd.getMonth());
    nd.setDate(newDim);
  }

  return nd;
}

// Next business day (weekend-aware, ignore holidays)
function nextBusinessDay(d: Date): Date {
  const nd = new Date(d);
  nd.setDate(nd.getDate() + 1);
  while (nd.getDay() === 0 || nd.getDay() === 6) {
    nd.setDate(nd.getDate() + 1);
  }
  return nd;
}

/* ---------- Cashflow & NSS helpers ---------- */

type Cashflow = {
  date: Date;
  tYears: number;
  amount: number; // per 100 notional
};

/**
 * Build all future cashflows (coupon + principal) from CURVE_DATE to maturity.
 * Works for notes/bonds; bills will be handled separately.
 */
function buildCashflowsForSecurity(params: {
  curveDate: Date;
  maturityDate: Date;
  interestRatePercent: number;
  frequencyPerYear: number;
  datedDate?: Date | null;
  firstInterestPaymentDate?: Date | null;
}): Cashflow[] {
  const {
    curveDate,
    maturityDate,
    interestRatePercent,
    frequencyPerYear,
    datedDate,
    firstInterestPaymentDate
  } = params;

  const faceValue = 100;
  const annualCouponRate = (interestRatePercent ?? 0) / 100;
  const couponPerPeriod = (faceValue * annualCouponRate) / frequencyPerYear;
  const monthsPerPeriod = 12 / frequencyPerYear;

  // Determine first coupon date
  let firstCouponDate: Date;

  if (firstInterestPaymentDate) {
    firstCouponDate = new Date(firstInterestPaymentDate);
  } else if (datedDate) {
    // Step forward from dated date until we pass it by one period
    let d = new Date(datedDate);
    firstCouponDate = addMonthsEOM(d, monthsPerPeriod);
  } else {
    // Fallback: approximate by stepping back from maturity
    let d = new Date(maturityDate);
    while (true) {
      const prev = addMonthsEOM(d, -monthsPerPeriod);
      if (prev <= CURVE_DATE || prev <= new Date(1970, 0, 1)) {
        firstCouponDate = d;
        break;
      }
      d = prev;
    }
  }

  const cashflows: Cashflow[] = [];

  // Generate coupon dates from firstCouponDate up to maturity
  let cfDate = new Date(firstCouponDate);
  while (cfDate <= maturityDate) {
    if (cfDate > curveDate) {
      const tYears = daysBetween(curveDate, cfDate) / DAYS_IN_YEAR;
      let amount = couponPerPeriod;
      if (cfDate.getTime() === maturityDate.getTime()) {
        amount += faceValue;
      }
      cashflows.push({ date: new Date(cfDate), tYears, amount });
    }
    cfDate = addMonthsEOM(cfDate, monthsPerPeriod);
  }

  return cashflows;
}

/**
 * NSS yield function:
 * params = [beta0, beta1, beta2, beta3, tau1, tau2]
 * returns annualized spot rate for tYears.
 */
function nssYield(tYears: number, params: number[]): number {
  const [b0, b1, b2, b3, tau1, tau2] = params;

  if (tYears <= 0) {
    return b0; // limit as t -> 0
  }

  const x1 = tYears / tau1;
  const x2 = tYears / tau2;

  const term1 = (1 - Math.exp(-x1)) / x1;
  const term2 = term1 - Math.exp(-x1);
  const term3 = (1 - Math.exp(-x2)) / x2 - Math.exp(-x2);

  return b0 + b1 * term1 + b2 * term2 + b3 * term3;
}

/**
 * Given dirty price and cashflows, compute the squared error
 * between market price and NSS theoretical price.
 */
function bondSquaredErrorFromNss(
  params: number[],
  bond: {
    dirtyPrice: number;
    cashflows: Cashflow[];
  }
): number {
  const { dirtyPrice, cashflows } = bond;

  let pvNss = 0;
  for (const cf of cashflows) {
    const y = nssYield(cf.tYears, params);
    const df = Math.exp(-y * cf.tYears);
    pvNss += cf.amount * df;
  }

  const err = dirtyPrice - pvNss;
  return err * err;
}

/**
 * Build coupon-bearing bonds (dirty price + cashflows) for NSS calibration.
 */
async function buildCouponBonds(env: Env): Promise<
  { cusip: string; dirtyPrice: number; cashflows: Cashflow[] }[]
> {
  const result = await env.DB.prepare(
    `
    SELECT
      p.CUSIP as cusip,
      p."END OF DAY" as dirty_price,
      p."MATURITY DATE" as price_maturity_date,
      s.IssueDate,
      s.SecurityType,
      s.SecurityTerm,
      s.MaturityDate,
      s.InterestRate,
      s.DatedDate,
      s.FirstInterestPaymentDate,
      s.InterestPaymentFrequency
    FROM nov18_treasury_securities p
    JOIN additional_data_new s
      ON p.CUSIP = s.CUSIP
    WHERE s.InterestRate IS NOT NULL
      AND s.InterestRate > 0
  `
  ).all<{
    cusip: string;
    dirty_price: number;
    price_maturity_date: string | null;
    IssueDate: string;
    SecurityType: string;
    SecurityTerm: string;
    MaturityDate: string;
    InterestRate: number;
    DatedDate: string;
    FirstInterestPaymentDate: string | null;
    InterestPaymentFrequency: string | null;
  }>();

  const rows = result.results ?? [];
  const bonds: { cusip: string; dirtyPrice: number; cashflows: Cashflow[] }[] =
    [];

  for (const r of rows) {
    const maturityDate = parseISODate(r.MaturityDate);
    const { frequency: frequencyPerYear, isBill } = getFrequencyPerYear(
      r.SecurityType,
      r.InterestPaymentFrequency,
      r.InterestRate ?? 0
    );

    if (isBill || frequencyPerYear === 0 || !r.InterestRate) {
      // Skip bills; NSS fit is focused on coupon-bearing bonds
      continue;
    }

    const cashflows = buildCashflowsForSecurity({
      curveDate: CURVE_DATE,
      maturityDate,
      interestRatePercent: r.InterestRate,
      frequencyPerYear,
      datedDate: r.DatedDate ? parseISODate(r.DatedDate) : null,
      firstInterestPaymentDate: r.FirstInterestPaymentDate
        ? parseISODate(r.FirstInterestPaymentDate)
        : null
    });

    if (cashflows.length === 0) continue;

    const dirtyPrice = Number(r.dirty_price);
    bonds.push({ cusip: r.cusip, dirtyPrice, cashflows });
  }

  return bonds;
}

/**
 * Calibrate NSS parameters by minimizing summed squared price errors
 * over all coupon-bearing bonds.
 */
async function calibrateNssFromDb(env: Env) {
  const bonds = await buildCouponBonds(env);

  if (bonds.length === 0) {
    return { error: "No bonds with usable coupon schedules were found for NSS fit." };
  }

  const loss = (theta: number[]) => {
    // theta: [b0, b1, b2, b3, ln_tau1, ln_tau2]
    const [b0, b1, b2, b3, lnTau1, lnTau2] = theta;
    const tau1 = Math.exp(lnTau1);
    const tau2 = Math.exp(lnTau2);
    const params = [b0, b1, b2, b3, tau1, tau2];

    let sse = 0;
    for (const bond of bonds) {
      sse += bondSquaredErrorFromNss(params, bond);
    }
    return sse;
  };

  const theta0 = [
    0.04, // beta0
    -0.02, // beta1
    0.02, // beta2
    0.01, // beta3
    Math.log(1.0), // tau1 ~ 1y
    Math.log(3.0) // tau2 ~ 3y
  ];

  const opt = nelderMead(loss, theta0);

  const [b0, b1, b2, b3, lnTau1, lnTau2] = opt.x;
  const tau1 = Math.exp(lnTau1);
  const tau2 = Math.exp(lnTau2);

  return {
    parameters: {
      beta0: b0,
      beta1: b1,
      beta2: b2,
      beta3: b3,
      tau1,
      tau2
    },
    sse: opt.fx,
    iterations: opt.iterations ?? undefined,
    bondsUsed: bonds.length
  };
}

/* ---------- Schedule tools (from starter) ---------- */

const scheduleTask = tool({
  description: "A tool to schedule a task to be executed at a later time",
  inputSchema: scheduleSchema,
  execute: async ({ when, description }) => {
    const { agent } = getCurrentAgent<Chat>();

    function throwError(msg: string): string {
      throw new Error(msg);
    }

    if (when.type === "no-schedule") {
      return "Not a valid schedule input";
    }

    const input =
      when.type === "scheduled"
        ? when.date
        : when.type === "delayed"
          ? when.delayInSeconds
          : when.type === "cron"
            ? when.cron
            : throwError("not a valid schedule input");

    try {
      agent!.schedule(input!, "executeTask", description);
    } catch (error) {
      console.error("error scheduling task", error);
      return `Error scheduling task: ${error}`;
    }
    return `Task scheduled for type "${when.type}" : ${input}`;
  }
});

const getScheduledTasks = tool({
  description: "List all tasks that have been scheduled",
  inputSchema: z.object({}),
  execute: async () => {
    const { agent } = getCurrentAgent<Chat>();

    try {
      const tasks = agent!.getSchedules();
      if (!tasks || tasks.length === 0) {
        return "No scheduled tasks found.";
      }
      return tasks;
    } catch (error) {
      console.error("Error listing scheduled tasks", error);
      return `Error listing scheduled tasks: ${error}`;
    }
  }
});

const cancelScheduledTask = tool({
  description: "Cancel a scheduled task using its ID",
  inputSchema: z.object({
    taskId: z.string().describe("The ID of the task to cancel")
  }),
  execute: async ({ taskId }) => {
    const { agent } = getCurrentAgent<Chat>();
    try {
      await agent!.cancelSchedule(taskId);
      return `Task ${taskId} has been successfully canceled.`;
    } catch (error) {
      console.error("Error canceling scheduled task", error);
      return `Error canceling task ${taskId}: ${error}`;
    }
  }
});

/* ---------- Treasury analytics tool ---------- */

const calculateTreasuryAnalytics = tool({
  description:
    'Look up a Treasury security by CUSIP in the D1 database tables "nov18_treasury_securities" and "additional_data_new", then compute clean price, accrued interest, dirty price, and the time-offset f (days into period / days in period).',
  inputSchema: z.object({
    cusip: z
      .string()
      .min(1)
      .describe("The 9-character CUSIP of the Treasury security.")
  })
});

/* ---------- NSS tools (schemas) ---------- */

const fitNssCurve = tool({
  description:
    "Fit a Nelson–Siegel–Svensson (NSS) yield curve to all coupon-bearing Treasuries as of Nov 19, 2025.",
  inputSchema: z.object({})
});

const analyzeBondsWithNss = tool({
  description:
    "Fit an NSS curve and compare NSS theoretical prices to dirty prices for all coupon-bearing Treasuries.",
  inputSchema: z.object({})
});

const analyzeCusipWithNss = tool({
  description:
    "For a given Treasury CUSIP, use the NSS curve to compute theoretical price, compare to dirty price, and report the error.",
  inputSchema: z.object({
    cusip: z
      .string()
      .min(1)
      .describe("The 9-character CUSIP of the Treasury security.")
  })
});

/* ---------- Export tool set ---------- */

export const tools = {
  scheduleTask,
  getScheduledTasks,
  cancelScheduledTask,
  calculateTreasuryAnalytics,
  fitNssCurve,
  analyzeBondsWithNss,
  analyzeCusipWithNss
} satisfies ToolSet;

/* ---------- Execution logic for tools that run on the server ---------- */

interface ToolContext {
  env: Env;
}

export const executions = {
  calculateTreasuryAnalytics: async (
    { cusip }: { cusip: string },
    { env }: ToolContext
  ) => {
    console.log("Running calculateTreasuryAnalytics for CUSIP:", cusip);

    try {
      // 1) Price row from nov18_treasury_securities (Nov 18, 2025 prices)
      const priceRow = await env.DB.prepare(
        `
        SELECT
          CUSIP as cusip,
          "END OF DAY" as clean_price,
          BUY as buy_price,
          SELL as sell_price,
          RATE as rate,
          "MATURITY DATE" as price_maturity_date
        FROM nov18_treasury_securities
        WHERE CUSIP = ?
        LIMIT 1
      `
      )
        .bind(cusip)
        .first<{
          cusip: string;
          clean_price: number;
          buy_price: number | null;
          sell_price: number | null;
          rate: number | null;
          price_maturity_date: string | null;
        }>();

      if (!priceRow) {
        return {
          error: `No price record found in 'nov18_treasury_securities' for CUSIP ${cusip}.`
        };
      }

      // 2) Security details from additional_data_new
      const secRow = await env.DB.prepare(
        `
        SELECT
          CUSIP as cusip,
          IssueDate,
          SecurityType,
          SecurityTerm,
          MaturityDate,
          InterestRate,
          DatedDate,
          FirstInterestPaymentDate,
          InterestPaymentFrequency
        FROM additional_data_new
        WHERE CUSIP = ?
        LIMIT 1
      `
      )
        .bind(cusip)
        .first<{
          cusip: string;
          IssueDate: string;
          SecurityType: string;
          SecurityTerm: string;
          MaturityDate: string;
          InterestRate: number;
          DatedDate: string;
          FirstInterestPaymentDate: string | null;
          InterestPaymentFrequency: string | null;
        }>();

      if (!secRow) {
        return {
          error: `No security record found in 'additional_data_new' for CUSIP ${cusip}.`
        };
      }

      // === Raw values ===
      const cleanPrice = Number(priceRow.clean_price); // per 100 face
      const datasetPriceDate = "2025-11-18"; // CSV date
      const maturityDate = parseISODate(secRow.MaturityDate);
      const annualCouponRate = (secRow.InterestRate ?? 0) / 100; // % -> decimal

      const { frequency: frequencyPerYear, isBill } = getFrequencyPerYear(
        secRow.SecurityType,
        secRow.InterestPaymentFrequency,
        secRow.InterestRate ?? 0
      );

      const faceValue = 100;

      // Settlement date: next business day after **today**
      const today = new Date();
      const settlementDate = nextBusinessDay(today);

      if (isBill || frequencyPerYear === 0 || annualCouponRate === 0) {
        // Bills / zero-coupon: no accrued interest
        return {
          cusip,
          securityType: secRow.SecurityType,
          securityTerm: secRow.SecurityTerm,
          datasetPriceDate,
          settlementDate: settlementDate.toISOString().slice(0, 10),
          issueDate: secRow.IssueDate,
          maturityDate: secRow.MaturityDate,
          couponRateAnnualPercent: secRow.InterestRate,
          cleanPricePer100: cleanPrice,
          accruedInterestPer100: 0,
          dirtyPricePer100: cleanPrice,
          f: 0,
          dayCountDetails: {
            convention: "Actual/Actual",
            isBill: true,
            daysInPeriod: 0,
            daysIntoPeriod: 0
          },
          explanation:
            "This security is treated as a bill (zero-coupon). There are no coupon payments, so accrued interest is zero and the dirty price equals the clean price. The time offset f is 0."
        };
      }

      // Notes / Bonds with coupons
      const couponPerPeriod =
        (faceValue * annualCouponRate) / frequencyPerYear;

      // Coupon dates around settlement (handles leap years via EOM logic)
      const monthsPerPeriod = 12 / frequencyPerYear;
      const { last: lastCouponDate, next: nextCouponDate } = ((): {
        last: Date;
        next: Date;
      } => {
        let current = new Date(maturityDate);
        let previous: Date;

        while (true) {
          previous = addMonthsEOM(current, -monthsPerPeriod);
          if (previous <= settlementDate || previous <= new Date(1970, 0, 1)) {
            break;
          }
          current = previous;
        }

        const last = previous <= settlementDate ? previous : current;
        const next = addMonthsEOM(last, monthsPerPeriod);
        return { last, next };
      })();

      const daysInPeriod = daysBetween(lastCouponDate, nextCouponDate);
      const daysIntoPeriod = daysBetween(lastCouponDate, settlementDate);
      const f = daysInPeriod === 0 ? 0 : daysIntoPeriod / daysInPeriod;

      const accruedInterestPer100 = couponPerPeriod * f;
      const dirtyPricePer100 = cleanPrice + accruedInterestPer100;

      return {
        cusip,
        securityType: secRow.SecurityType,
        securityTerm: secRow.SecurityTerm,
        datasetPriceDate,
        settlementDate: settlementDate.toISOString().slice(0, 10),
        issueDate: secRow.IssueDate,
        maturityDate: secRow.MaturityDate,
        couponRateAnnualPercent: secRow.InterestRate,
        couponPerPeriodPer100: couponPerPeriod,
        cleanPricePer100: cleanPrice,
        accruedInterestPer100,
        dirtyPricePer100,
        f,
        dayCountDetails: {
          convention: "Actual/Actual with end-of-month adjustment",
          lastCouponDate: lastCouponDate.toISOString().slice(0, 10),
          nextCouponDate: nextCouponDate.toISOString().slice(0, 10),
          daysInPeriod,
          daysIntoPeriod
        },
        explanation:
          "Using the Nov 18, 2025 clean price from 'nov18_treasury_securities', we treat settlement as the next business day after today. The coupon schedule comes from 'additional_data_new', with an end-of-month rule so a Feb 28 coupon becomes Feb 29 in leap years. We compute f = daysIntoPeriod / daysInPeriod, accrued interest = couponPerPeriod * f, and dirty price = clean price + accrued interest."
      };
    } catch (err) {
      console.error("Error in calculateTreasuryAnalytics:", err);
      return {
        error:
          "An internal error occurred while calculating treasury analytics. Check the worker logs for details."
      };
    }
  },

  fitNssCurve: async (_args: {}, { env }: ToolContext) => {
    console.log("Running fitNssCurve over all coupon-bearing securities");
    const result = await calibrateNssFromDb(env);
    return result;
  },

  analyzeBondsWithNss: async (_args: {}, { env }: ToolContext) => {
    console.log("Running analyzeBondsWithNss (fit NSS + compare prices)");

    const fitResult = await calibrateNssFromDb(env);
    if ((fitResult as any).error) {
      return fitResult;
    }

    const { beta0, beta1, beta2, beta3, tau1, tau2 } =
      fitResult.parameters as {
        beta0: number;
        beta1: number;
        beta2: number;
        beta3: number;
        tau1: number;
        tau2: number;
      };

    const params = [beta0, beta1, beta2, beta3, tau1, tau2];
    const bonds = await buildCouponBonds(env);

    const perBond: {
      cusip: string;
      dirtyPrice: number;
      nssPrice: number;
      error: number;
      errorSquared: number;
    }[] = [];

    let sse = 0;

    for (const bond of bonds) {
      let nssPrice = 0;
      for (const cf of bond.cashflows) {
        const y = nssYield(cf.tYears, params);
        const df = Math.exp(-y * cf.tYears);
        nssPrice += cf.amount * df;
      }

      const error = bond.dirtyPrice - nssPrice;
      const errorSquared = error * error;
      sse += errorSquared;

      perBond.push({
        cusip: bond.cusip,
        dirtyPrice: bond.dirtyPrice,
        nssPrice,
        error,
        errorSquared
      });
    }

    const n = perBond.length || 1;
    const rmse = Math.sqrt(sse / n);

    return {
      curveDate: CURVE_DATE.toISOString().slice(0, 10),
      nssParameters: {
        beta0,
        beta1,
        beta2,
        beta3,
        tau1,
        tau2
      },
      summary: {
        bondsAnalyzed: perBond.length,
        sse,
        rmse,
        calibrationSse: fitResult.sse,
        calibrationBondsUsed: fitResult.bondsUsed
      },
      perBond
    };
  },

  analyzeCusipWithNss: async (
    { cusip }: { cusip: string },
    { env }: ToolContext
  ) => {
    console.log("Running analyzeCusipWithNss for CUSIP:", cusip);

    const nss = await calibrateNssFromDb(env);
    if ((nss as any).error) {
      return nss;
    }

    const { beta0, beta1, beta2, beta3, tau1, tau2 } = nss.parameters;
    const params = [beta0, beta1, beta2, beta3, tau1, tau2];

    // 2) Pull this CUSIP's dirty price + security info
    const priceRow = await env.DB.prepare(
      `
      SELECT
        CUSIP as cusip,
        "END OF DAY" as dirty_price,
        BUY as buy_price,
        SELL as sell_price,
        RATE as coupon_rate,
        "MATURITY DATE" as maturity_date
      FROM nov18_treasury_securities
      WHERE CUSIP = ?
      LIMIT 1
    `
    )
      .bind(cusip)
      .first<{
        cusip: string;
        dirty_price: number;
        buy_price: number | null;
        sell_price: number | null;
        coupon_rate: number | null;
        maturity_date: string;
      }>();

    if (!priceRow) {
      return {
        error: `No price record found in 'nov18_treasury_securities' for CUSIP ${cusip}.`
      };
    }

    const secRow = await env.DB.prepare(
      `
      SELECT
        CUSIP as cusip,
        IssueDate,
        SecurityType,
        SecurityTerm,
        MaturityDate,
        InterestRate,
        DatedDate,
        FirstInterestPaymentDate,
        InterestPaymentFrequency
      FROM additional_data_new
      WHERE CUSIP = ?
      LIMIT 1
    `
    )
      .bind(cusip)
      .first<{
        cusip: string;
        IssueDate: string;
        SecurityType: string;
        SecurityTerm: string;
        MaturityDate: string;
        InterestRate: number;
        DatedDate: string;
        FirstInterestPaymentDate: string | null;
        InterestPaymentFrequency: string | null;
      }>();

    if (!secRow) {
      return {
        error: `No security record found in 'additional_data_new' for CUSIP ${cusip}.`
      };
    }

    const face = 100;
    const annualCoupon = (secRow.InterestRate ?? 0) / 100;
    const { frequency, isBill } = getFrequencyPerYear(
      secRow.SecurityType,
      secRow.InterestPaymentFrequency,
      secRow.InterestRate ?? 0
    );

    const dirtyPrice = Number(priceRow.dirty_price);

    if (isBill || frequency === 0 || annualCoupon === 0) {
      // For bills we just compare clean/discounted single CF
      const maturityDate = new Date(secRow.MaturityDate);
      const tYears =
        (maturityDate.getTime() - CURVE_DATE.getTime()) /
        (DAYS_IN_YEAR * 24 * 3600 * 1000);

      const ySpot = nssYield(tYears, params);
      const df = Math.exp(-ySpot * tYears);
      const theoreticalPrice = face * df;

      return {
        cusip,
        type: secRow.SecurityType,
        isBill: true,
        nssParams: nss.parameters,
        datasetDate: CURVE_DATE.toISOString().slice(0, 10),
        dirtyPricePer100: dirtyPrice,
        theoreticalPricePer100: theoreticalPrice,
        pricingError: theoreticalPrice - dirtyPrice,
        explanation:
          "For a bill, we treat the single face-value cash flow at maturity and discount it using the NSS spot rate."
      };
    }

    // 3) Build future coupon schedule from CURVE_DATE to maturity
    const maturityDate = new Date(secRow.MaturityDate);
    const cashflowsBasic = buildCashflowsForSecurity({
      curveDate: CURVE_DATE,
      maturityDate,
      interestRatePercent: secRow.InterestRate,
      frequencyPerYear: frequency,
      datedDate: secRow.DatedDate ? parseISODate(secRow.DatedDate) : null,
      firstInterestPaymentDate: secRow.FirstInterestPaymentDate
        ? parseISODate(secRow.FirstInterestPaymentDate)
        : null
    });

    const cashflows = cashflowsBasic.map((cf) => {
      const ySpot = nssYield(cf.tYears, params);
      const df = Math.exp(-ySpot * cf.tYears);
      const pv = cf.amount * df;
      return {
        date: cf.date.toISOString().slice(0, 10),
        tYears: cf.tYears,
        amount: cf.amount,
        spotRate: ySpot,
        discountFactor: df,
        presentValue: pv
      };
    });

    const theoreticalPrice = cashflows.reduce(
      (sum, cf) => sum + cf.presentValue,
      0
    );
    const pricingError = theoreticalPrice - dirtyPrice;

    return {
      cusip,
      type: secRow.SecurityType,
      term: secRow.SecurityTerm,
      nssParams: nss.parameters,
      nssFitSse: nss.sse,
      datasetDate: CURVE_DATE.toISOString().slice(0, 10),
      dirtyPricePer100: dirtyPrice,
      theoreticalPricePer100: theoreticalPrice,
      pricingError,
      cashflows,
      explanation:
        "We fit a global NSS curve on all coupon-bearing Treasuries as of Nov 19, 2025, then discounted each future coupon and principal cash flow for this CUSIP using the NSS spot rate at its time-to-maturity. The sum of discounted cash flows gives the theoretical NSS price, which we compare to the market (dirty) price."
    };
  }
};
