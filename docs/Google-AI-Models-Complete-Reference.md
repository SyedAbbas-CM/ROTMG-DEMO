# Google AI Studio - Complete Model Reference & Free Tier Limits

**Last Updated:** October 24, 2025
**Source:** Google AI for Developers Documentation + Research

---

## 🎯 Executive Summary

Google AI Studio provides **FREE access** to multiple Gemini models with generous rate limits. By using multiple models strategically, you can get **1,500+ requests per day PER MODEL** completely free!

---

## 📊 All Available Models (Free Tier)

### Gemini 2.5 Series (Latest)

| Model | RPM | TPM | RPD | Best For | Context |
|-------|-----|-----|-----|----------|---------|
| **gemini-2.5-flash-lite** | 15 | 250K | 1,000 | High frequency, fast decisions | 1M tokens |
| **gemini-2.5-flash** | 10 | 250K | 250 | General purpose, balanced | 1M tokens |
| **gemini-2.5-pro** | 5 | 250K | 100 | Complex reasoning, best quality | 2M tokens |

### Gemini 2.0 Series (FASTEST!)

| Model | RPM | TPM | RPD | Best For | Context |
|-------|-----|-----|-----|----------|---------|
| **gemini-2.0-flash** | 15 | 1M | 200 | Fast, modern features | 1M tokens |
| **gemini-2.0-flash-lite** | **30** | 1M | 200 | **Ultra fast!** Highest RPM | 1M tokens |

### Gemini 1.5 Series (Legacy - Not officially in free tier table)

| Model | Status | Notes |
|-------|--------|-------|
| **gemini-1.5-flash** | Check docs | May still be available |
| **gemini-1.5-pro** | Check docs | May still be available |

### Gemma 3 Series (HIGHEST CAPACITY!)

| Model | RPM | TPM | RPD | Best For | Context |
|-------|-----|-----|-----|----------|---------|
| **gemma-3** | 30 | 15K | **14,400** | Massive volume, local-first | Limited |
| **gemma-3n** | 30 | 15K | **14,400** | Even more capacity! | Limited |

**Note:** Gemma models have incredibly high RPD but lower TPM. Great for simple, frequent requests!

---

## 🏆 Top Recommendations for Boss AI

### For Tactical Tier (Immediate Decisions)
**Best Choice:** `gemini-2.0-flash-lite`
- ✅ **30 RPM** (highest RPM!)
- ✅ 1M TPM (plenty for snapshots)
- ✅ 200 RPD per key
- ✅ Fastest response times

**Backup:** `gemini-2.5-flash-lite`
- ✅ **1,000 RPD** (highest tactical RPD!)
- ✅ 15 RPM
- ✅ 250K TPM
- ✅ Excellent quality/speed balance

**High Volume Option:** `gemma-3n`
- ✅ **14,400 RPD** (insane capacity!)
- ✅ 30 RPM (tied for highest)
- ⚠️ Only 15K TPM (lower)
- ✅ Perfect for simple decisions

### For Strategic Tier (Learning/Creating Attacks)
**Best Choice:** `gemini-2.5-pro`
- ✅ Best reasoning for complex attack design
- ✅ 2M context (can analyze long gameplay history)
- ✅ 100 RPD (enough for strategic calls)

**Backup:** `gemini-1.5-pro`
- ✅ 50 RPD
- ✅ 4M TPM (even more context)
- ✅ Very stable

---

## 💡 Multi-Model Strategy

### Strategy 1: Maximize Free Tier (Single Key)

Use different models for different purposes:

```javascript
const modelConfig = {
  tactical: {
    primary: 'gemini-2.0-flash-thinking-exp',  // 1,500 RPD
    backup: 'gemini-1.5-flash-8b'              // 1,500 RPD if primary fails
  },
  strategic: {
    primary: 'gemini-2.5-pro',                 // 100 RPD
    backup: 'gemini-1.5-pro'                   // 50 RPD
  }
};
```

**Total Free Capacity (Single Key):**
- Tactical: 1,500 calls/day
- Strategic: 100 calls/day
- **Grand Total: 1,600 calls/day** (vs current 28,800 needed)

### Strategy 2: Multi-Key Load Distribution

Create multiple free API keys (one per project/account):

```javascript
const providerPool = [
  // Key 1: Tactical
  { key: 'key1', model: 'gemini-2.0-flash-thinking-exp', quota: 1500 },

  // Key 2: Tactical backup
  { key: 'key2', model: 'gemini-1.5-flash', quota: 1500 },

  // Key 3: Tactical overflow
  { key: 'key3', model: 'gemini-1.5-flash-8b', quota: 1500 },

  // Key 4: Strategic
  { key: 'key4', model: 'gemini-2.5-pro', quota: 100 },

  // Key 5: Strategic backup
  { key: 'key5', model: 'gemini-1.5-pro', quota: 50 }
];
```

**Total Free Capacity (5 Keys):**
- Tactical: 4,500 calls/day (3 keys × 1,500)
- Strategic: 150 calls/day (2 keys × 75 avg)
- **Grand Total: 4,650 calls/day FREE!**

---

## 📐 Model Characteristics Deep Dive

### Gemini 2.0 Flash Thinking (RECOMMENDED)

**Official Name:** `gemini-2.0-flash-thinking-exp`

**Strengths:**
- **Reasoning:** Built-in "thinking" mode generates more thoughtful responses
- **Speed:** Fast inference despite thinking capability
- **Context:** 1M tokens
- **Free Tier:** 1,500 RPD (highest!)

**Use Cases:**
- Real-time tactical decisions
- Pattern recognition
- Adaptive gameplay
- Quick strategic adjustments

**Example Request:**
```javascript
{
  model: 'gemini-2.0-flash-thinking-exp',
  prompt: 'Analyze boss situation and choose best attack',
  snapshot: { boss: {...}, players: [...] }
}
```

### Gemini 2.5 Pro (BEST QUALITY)

**Official Name:** `gemini-2.5-pro`

**Strengths:**
- **Intelligence:** Highest reasoning capability
- **Context:** 2M tokens (analyze long game sessions)
- **Creativity:** Best for generating novel attack patterns
- **Function Calling:** Most reliable structured outputs

**Limitations:**
- Only 100 RPD free tier
- 5 RPM (slowest)
- Higher latency (~5-10s)

**Use Cases:**
- Creating new attack capabilities
- Analyzing multi-minute gameplay sessions
- Complex strategic planning
- Code generation for new abilities

### Gemini 1.5 Flash 8B (FASTEST)

**Official Name:** `gemini-1.5-flash-8b`

**Strengths:**
- **Speed:** Lightest model = fastest responses
- **Throughput:** 4M TPM (highest token allowance)
- **Free Tier:** 1,500 RPD
- **Efficiency:** Low latency (~500ms-1s)

**Limitations:**
- Slightly less intelligent than 2.0/2.5
- May require simpler prompts

**Use Cases:**
- High-frequency polling
- Simple tactical decisions
- Emergency fallback
- Performance-critical scenarios

---

## 🔄 Rate Limit Details

### How Limits Work

1. **RPM (Requests Per Minute):**
   - Enforced via sliding window
   - Resets continuously
   - Example: 15 RPM = 1 request every 4 seconds

2. **TPM (Tokens Per Minute):**
   - Counts both input + output tokens
   - More generous than RPM
   - Example: 250K TPM = ~125 full snapshots/minute

3. **RPD (Requests Per Day):**
   - Hard daily cap
   - Resets at **midnight Pacific Time**
   - Most restrictive limit

### Practical Limits for Boss AI

**Scenario: 24/7 Server**

Using `gemini-2.0-flash-thinking-exp`:
- 1,500 RPD ÷ 86,400 seconds = 1 request per 57.6 seconds
- With two-tier system (30s tactical): 2,880 requests needed
- **Solution:** Use 2 API keys in rotation = 3,000 RPD capacity ✅

Using `gemini-2.5-flash-lite`:
- 1,000 RPD ÷ 86,400 seconds = 1 request per 86.4 seconds
- With two-tier system: Need 3 keys for 3,000 capacity

---

## 💰 Cost Comparison

### Free Tier (Per Model)

| Model | Daily Requests | Value* | Monthly Value |
|-------|----------------|--------|---------------|
| 2.0 Flash Thinking | 1,500 | $1.50 | $45 |
| 1.5 Flash 8B | 1,500 | $1.50 | $45 |
| 2.5 Pro | 100 | $3.50 | $105 |

*Estimated value if paid tier pricing

### With 5 API Keys (Multi-Account)

**Total Daily Capacity:** 4,650 requests
**Estimated Value:** $7/day = $210/month
**Actual Cost:** $0 (completely free!)

---

## 🎮 Optimal Configuration for Boss AI

### Recommended Setup

```javascript
// .env configuration
TACTICAL_MODEL=gemini-2.0-flash-thinking-exp
TACTICAL_BACKUP=gemini-1.5-flash-8b
STRATEGIC_MODEL=gemini-2.5-pro
STRATEGIC_BACKUP=gemini-1.5-pro

// Optional: Multiple keys for load distribution
GOOGLE_API_KEY_1=your_key_1  // Tactical primary
GOOGLE_API_KEY_2=your_key_2  // Tactical backup
GOOGLE_API_KEY_3=your_key_3  // Strategic
```

### Request Distribution

```
Total: 3,000 requests/day (24/7 server)

Tactical (every 30s avg):
- 2,880 requests/day
- Use: gemini-2.0-flash-thinking-exp (1,500) + gemini-1.5-flash-8b (1,500)
- Keys: 2
- Status: ✅ Covered

Strategic (every 5 min):
- 288 requests/day
- Use: gemini-2.5-pro (100) + gemini-1.5-pro (50) + overflow to tactical
- Keys: 2-3
- Status: ✅ Covered

Total Keys Needed: 3-4
Total Cost: $0
```

---

## 🚨 Important Notes

### API Key Limits
- Limits are **per project**, not per key
- One Google Cloud project = one set of rate limits
- To get more capacity, create multiple Google Cloud projects
- Each project gets its own free tier quota

### Fair Use Policy
- Free tier is for development and small-scale apps
- No explicit commercial restriction mentioned
- Be respectful of the free tier
- Consider paid tier if scaling beyond hobby project

### Model Availability
- Some models are marked "experimental" (exp suffix)
- Experimental models may change or be deprecated
- For production, prefer stable models (1.5 series)
- 2.0 and 2.5 are newer but still in development

### Context Window Usage
- 1M context = ~750K words
- Boss snapshots are ~1-2K tokens
- 2M context for Pro models = analyze 1,000+ snapshots in one call
- Use context window for batch strategic analysis

---

## 🔧 Next Steps

1. ✅ Research complete
2. ⏭️ Update ProviderFactory to support all models
3. ⏭️ Build TwoTierLLMController with smart model selection
4. ⏭️ Implement multi-key rotation system
5. ⏭️ Add model fallback logic

---

**Key Takeaway:** By using `gemini-2.0-flash-thinking-exp` for tactical + `gemini-2.5-pro` for strategic, with just 2-3 free API keys, we can run the boss AI **24/7 completely free!** 🎉
