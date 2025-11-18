# Database Connection Issue - 2025-11-18

## Issue
Direct PostgreSQL connection to Supabase is blocked with "No route to host" error.

## Current Configuration
```env
DATABASE_URL=postgresql://postgres:LMd%5Eo0rNDU6r7sgP@db.dzlxobcplcaiijkddrvd.supabase.co:5432/postgres
```

## Error
```
OSError: [Errno 65] No route to host
```

## Root Cause
Supabase free tier often restricts direct external connections to port 5432 for security. The database is accessible via REST API but not direct PostgreSQL connection.

## Verified Working
- ✅ REST API: `https://dzlxobcplcaiijkddrvd.supabase.co` (HTTP 200)
- ✅ Supabase Client: Connected successfully via `supabase-py`
- ❌ Direct PostgreSQL: Blocked (asyncpg cannot connect)

## Solutions

### Option 1: Connection Pooler (Recommended)
If available in Supabase dashboard:
1. Go to Project Settings → Database
2. Look for "Connection pooling" section
3. Get connection string like:
   ```
   postgresql://postgres.dzlxobcplcaiijkddrvd:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
   ```
4. Update `DATABASE_URL` in `.env`
5. This works with existing `database.py` (asyncpg)

### Option 2: Supabase Client (Alternative)
Modify `conduit/core/database.py` to use Supabase Python client instead of asyncpg:
- Already tested and working ✅
- Uses REST API instead of direct SQL
- Trade-off: Slightly higher latency, but works with current free tier

### Option 3: Database Paused
Check Supabase dashboard for "Database Paused" status and resume if needed.

## Current Status
**Phase 1**: Database code is written (`database.py`) but untested due to connection restriction.

**Phase 2**: Will need working database connection for:
- Saving queries and routing decisions
- Storing feedback
- Updating model states (Thompson Sampling parameters)

## Recommendation
For MVP, we can:
1. **Short-term**: Use in-memory storage for testing Phase 1 routing logic
2. **Production**: Get pooler connection string or upgrade Supabase tier

## Next Steps
- [ ] Check Supabase dashboard for pooler connection string
- [ ] Or consider using Supabase client for database operations
- [ ] Document final approach in `database.py` once connection is resolved
