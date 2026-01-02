# Lag Fix Action Plan - Executive Summary

**Created**: 2025-11-02
**Status**: Ready for Implementation
**Target**: Make game playable at 200-300ms ping

---

## Problem Statement

Game is **unplayable at 200ms ping**:
- ❌ Chunks don't load (1.9s delay, movement blocked)
- ❌ Shooting unreliable (~85% success rate)
- ❌ Entities teleport (no interpolation)
- ❌ Hit detection feels unfair (no lag compensation)

---

## Root Causes Identified

### 1. Client Missing worldId in BULLET_CREATE ⚠️ **BUG**
```javascript
// current: input.js
networkManager.sendShoot({ x, y, angle, speed, damage });  // ❌ Missing worldId
```
**Impact**: Server rejects bullets without worldId → silent failures

### 2. Chunk Throttle Too Aggressive
```javascript
// current: 1500ms fixed throttle
if (now - lastReq >= 1500) { requestChunk(); }
```
**Impact**: At 200ms ping, 400ms round-trip + 1500ms throttle = 1.9s total

### 3. Unloaded Chunks Block Movement
```javascript
// current: ClientMapManager.js:946
if (!tile) return true;  // ❌ Blocks player
```
**Impact**: Player stutters every time entering unexplored areas

### 4. No Retry Mechanism
**Impact**: Lost packets = lost bullets, stuck chunk requests

### 5. No Interpolation
**Impact**: Entities teleport instead of moving smoothly

---

## Good News: Strong Foundation

✅ **Server already at 30 FPS** (not 10 FPS)
✅ **WebSocket compression enabled** (60% bandwidth reduction)
✅ **Visibility culling working** (40-tile radius)
✅ **Collision validation sophisticated** (dedup, timestamp, LOS checks)
✅ **Data structures optimized** (Structure of Arrays, typed arrays)
✅ **Bullet spawn offset** (prevents wall collision)

**Conclusion**: Core architecture is solid, just need reliability & smoothness layers!

---

## Three-Tiered Plan

### 🚀 Tier 1: Emergency Fixes (2 hours - DO TODAY)

**Goal**: Fix critical bug, improve reliability immediately

| Fix | File | Change | Impact |
|-----|------|--------|--------|
| **Add worldId** | `public/src/game/input.js:58` | Add `worldId: gameState.character.worldId` to sendShoot() | Fix silent bullet failures |
| **Extend validation window** | `src/entities/CollisionManager.js:267` | Change 500ms → 1000ms | Prevent false collision rejections |
| **Test with latency** | Terminal | `tc qdisc add dev lo root netem delay 200ms` (Linux) | Validate fixes work |

**Expected Result**: 5-10% better shot reliability today

---

### 🔧 Tier 2: Reliability Layer (1 week - ~15 hours)

**Goal**: Make game playable at 200ms ping

#### Part A: Chunk Loading Fixes (4 hours)
1. **Adaptive throttling** (2 hours)
   - Replace fixed 1500ms with `max(500, RTT × 2)`
   - At 200ms ping: 400ms throttle instead of 1500ms

2. **Remove movement blocking** (1 hour)
   - Change `isWallOrObstacle` to allow movement through unloaded chunks
   - Request urgent chunk when player enters unloaded area

3. **Timeout & retry** (1 hour)
   - Add ChunkRequestManager with timeout after RTT × 3
   - Retry up to 5 times with exponential backoff

**Expected Result**: Chunks load in < 800ms (vs 1900ms current), no stuttering

#### Part B: Shooting Reliability (6 hours)
1. **RTT measurement** (2 hours)
   - Implement PING/PONG system
   - Calculate exponential moving average
   - Display in UI

2. **Bullet acknowledgment** (4 hours)
   - Add `BULLET_ACK` message type to protocol
   - Client tracks pending bullets (Map<requestId, {data, sentAt, retries}>)
   - Server sends immediate ACK with acceptance/rejection
   - Client retries on timeout (RTT × 3), max 3 retries

**Expected Result**: 99%+ shot reliability

#### Part C: User Feedback (2 hours)
- Network stats panel (RTT, jitter, packet loss estimate)
- "Chunk loading..." indicators
- "Shot rejected" notifications
- Connection quality indicator (green/yellow/red)

**Expected Result**: Users understand what's happening

---

### ✨ Tier 3: Smoothness Layer (2 weeks - ~25 hours)

**Goal**: Make game feel smooth at 200-300ms ping

#### Week 1: Interpolation (10 hours)
1. **EntityInterpolator class** (6 hours)
   - Buffer entity snapshots (last 500ms)
   - Render 100-150ms in the past
   - Linear interpolation between snapshots
   - Skip local player (use client prediction)

2. **Integration** (4 hours)
   - Integrate into game render loop
   - Handle edge cases (new entities, removed entities)
   - Tune buffer time based on measured jitter

**Expected Result**: Smooth 60 FPS rendering from 30 FPS updates

#### Week 2: Optimization (8 hours)
1. **Delta compression** (5 hours)
   - Track last sent state per client
   - Send only changed fields
   - Full sync every 10 seconds
   - ~75% bandwidth reduction

2. **Adaptive update rates** (3 hours)
   - Client sends at variable rate (16-100ms based on RTT)
   - Immediate updates for critical actions (shooting)
   - ~50% outgoing bandwidth reduction

**Expected Result**: Bandwidth reduced from 600 KB/s → 150 KB/s

#### Optional: Lag Compensation (7 hours)
1. **Position history** (4 hours)
   - Record entity positions every 50ms
   - Keep 1 second of history
   - Rewind entities to client's perceived time

2. **Collision rewinding** (3 hours)
   - On collision validation, rewind enemy to RTT/2 ago
   - Check collision with rewound position
   - Industry standard (CS:GO, Overwatch)

**Expected Result**: Fair hit detection for high-latency players

---

## Recommended Approach

### Option A: Incremental (Recommended)
**Timeline**: 3-4 weeks
**Risk**: Low
**Learning**: High

```
Week 1: Tier 1 (2h) + Tier 2 Part A+B (10h) = 12h
Week 2: Tier 2 Part C (2h) + Tier 3 Week 1 (10h) = 12h
Week 3: Tier 3 Week 2 (8h) + Testing (4h) = 12h
Week 4: Optional lag compensation (7h) + Polish (5h) = 12h
```

**Benefit**: Can test and validate each layer before moving to next

### Option B: MVP (Fast Track)
**Timeline**: 1 week
**Risk**: Medium
**Focus**: Critical fixes only

```
Day 1: Tier 1 (2h)
Day 2-3: Tier 2 Part A+B (10h)
Day 4: Tier 2 Part C + Testing (4h)
Day 5: Polish + Documentation (4h)
```

**Benefit**: Game playable ASAP, can add smoothness later

### Option C: Comprehensive (Best Quality)
**Timeline**: 6 weeks
**Risk**: Low
**Quality**: Production-ready

```
Week 1: Tier 1 + Planning
Week 2: Tier 2 Part A+B
Week 3: Tier 2 Part C + Tier 3 Week 1
Week 4: Tier 3 Week 2
Week 5: Optional lag compensation
Week 6: Testing, profiling, optimization
```

**Benefit**: Fully polished, tested, documented

---

## Files to Modify

### Tier 1 (Emergency)
- `public/src/game/input.js` (add worldId)
- `src/entities/CollisionManager.js` (extend timeout)

### Tier 2 (Reliability)
- `public/src/network/ClientNetworkManager.js` (RTT, ACK tracking, retry)
- `public/src/map/ClientMapManager.js` (adaptive throttle, remove blocking, timeout)
- `common/protocol.js` (add message types: BULLET_ACK, CHUNK_TIMEOUT)
- `Server.js` (handle BULLET_ACK, send acknowledgments)
- `public/src/ui/NetworkStats.js` (NEW - stats panel)

### Tier 3 (Smoothness)
- `public/src/network/EntityInterpolator.js` (NEW - interpolation)
- `public/src/network/DeltaCompressor.js` (NEW - delta encoding)
- `public/src/game/game.js` (integrate interpolator)
- `Server.js` (delta compression on broadcast)
- `src/network/LagCompensator.js` (NEW - optional, lag compensation)

**Total**: 7 files modified, 4 new files (or 5 with lag compensation)

---

## Testing Strategy

### Local Testing (Simple)
```bash
# Simulate 200ms latency on localhost (Linux)
sudo tc qdisc add dev lo root netem delay 100ms

# Or macOS (requires Network Link Conditioner)
# System Preferences → Developer Tools → Network Link Conditioner
```

### Remote Testing (Real-world)
1. Deploy to Windows laptop via PlayIt.gg tunnel + test from external network
2. Use VPN to simulate distance/latency
3. Test from different geographic locations via friends/testers

### Load Testing
1. Run 10 concurrent clients with different latencies
2. Monitor bandwidth, CPU, memory
3. Measure success rates (chunks loaded, bullets created, collisions validated)

---

## Success Criteria

### Tier 1 Success
- ✅ worldId included in all bullet messages
- ✅ No console errors about rejected bullets
- ✅ Collisions validated at 200ms+ latency

### Tier 2 Success (Playability)
- ✅ Chunks load in < 800ms at 200ms ping
- ✅ Player can move without stuttering
- ✅ 99%+ bullets reach server and are confirmed
- ✅ RTT displayed in UI
- ✅ Users see loading feedback

### Tier 3 Success (Smoothness)
- ✅ Entities move smoothly at 60 FPS (no teleporting)
- ✅ Bandwidth reduced by 70%+
- ✅ Hit detection feels fair
- ✅ Game playable at 300ms ping

---

## Cost-Benefit Analysis

### Time Investment
- **Tier 1**: 2 hours → Game functional
- **Tier 2**: 15 hours → Game playable
- **Tier 3**: 25 hours → Game smooth

**Total**: ~40 hours over 3-4 weeks

### Value Gained
- Support cross-continent players (US ↔ EU, ~150ms)
- Support mobile connections (150-300ms typical)
- Support budget hosting (single server globally)
- Reduce packet loss impact (retry logic)
- Professional-grade feel (interpolation, lag comp)

### Alternative Cost
- **Multiple regional servers**: $50-200/month × 3 regions = $150-600/month
- **CDN for game state**: Not feasible (need stateful server)
- **Accept limited audience**: Lose 50%+ of potential players

**ROI**: 40 hours investment → Unlock global audience + reduce hosting costs

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tier 1 breaks existing code | Low | High | Test on dev server first, git branch |
| RTT measurement overhead | Low | Low | Send PING every 5 seconds (negligible) |
| Interpolation feels wrong | Medium | Medium | Make buffer time configurable, tune based on testing |
| Delta compression bugs | Medium | High | Add full sync fallback every 10s, extensive testing |
| Server CPU increase | Medium | Medium | Profile before/after, optimize hot paths |
| Players abuse retry system | Low | Low | Rate limit per client (max 10 retries/sec) |

**Overall Risk**: **Low-Medium** (mostly additive changes to solid foundation)

---

## Decision Point

**Which approach should we take?**

1. **Option A (Incremental)** - Safest, best for learning ⭐ Recommended
2. **Option B (MVP)** - Fastest to playable state
3. **Option C (Comprehensive)** - Best quality, most time

**Next Steps**:
1. Choose approach
2. Start with Tier 1 (2 hours)
3. Test with simulated latency
4. If successful, proceed to Tier 2

---

## Questions to Answer

1. **Do we want to support 300ms+ ping?** (affects buffer sizes, timeouts)
2. **What's the max bandwidth per player?** (affects delta compression priority)
3. **Do we plan multiple servers eventually?** (affects architecture decisions)
4. **What's acceptable server CPU increase?** (affects tick rate, history storage)
5. **Is mobile support important?** (affects UI feedback, adaptive rates)

**Default Assumptions** (if not specified):
- Target: 200-300ms ping
- Bandwidth: < 200 KB/s per player
- Single server for now (regional later)
- CPU: +20% acceptable
- Mobile: Nice to have

---

## Summary

**Problem**: Game unplayable at 200ms+ ping
**Root Cause**: Missing reliability & smoothness layers
**Solution**: 3-tier incremental improvements
**Time**: 2 hours → 1 week → 3-4 weeks (pick your level)
**Risk**: Low (building on solid foundation)
**Outcome**: Global playability, professional feel

**Recommendation**: Start with Tier 1 today (2 hours), validate, then proceed to Tier 2.

Ready to begin? 🚀
