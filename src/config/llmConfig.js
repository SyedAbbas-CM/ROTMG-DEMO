export default {
  // Seconds between consecutive LLM calls if the snapshot changed.
  planPeriodSec: +process.env.LLM_PLAN_PERIOD || 2,
  // Cool-down applied after any error / timeout from the provider.
  backoffSec: +process.env.LLM_BACKOFF_SEC || 15,
  // Seconds between boss taunts / speech lines
  speechPeriodSec: +process.env.LLM_SPEECH_PERIOD || 6
}; 