# Next Steps & Future Phases Guide

## Immediate Actions (This Week)

### 1. Validate Database Migration
```bash
# Run migration in development
cd backend
alembic upgrade head

# Verify tables created
psql -c "\dt session_*"
psql -c "\d session_access"
psql -c "\d session_access_origins"
psql -c "\d session_group_links"
```

**Checklist:**
- [ ] All tables exist with correct columns
- [ ] Unique constraints enforced
- [ ] Indexes created
- [ ] Foreign keys functional
- [ ] Default values work (server_default for booleans)

### 2. Run Backend Tests
```bash
# Unit tests for new services
pytest tests/services/test_session_access_service.py -v

# API integration tests
pytest tests/api/v1/test_session_access.py -v
pytest tests/api/v1/test_session_user_state.py -v

# Authorization tests
pytest tests/core/test_authorization.py -v
```

**Test Coverage Needed:**
- Direct access grant/revoke
- Group linkage and member sync
- Access origin tracking
- Cascade delete behavior
- User state transitions
- Authorization checks

### 3. Manual API Testing
```bash
# Start dev server
python -m uvicorn app.main:app --reload

# Test endpoints with curl or Postman
# Create routine (auto-creates session)
curl -X POST http://localhost:8000/api/v1/routines \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Routine", "dance_id": "..."}'

# Grant access to user
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/access/users \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"user_id": "...", "role": "admin"}'

# Link group
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/groups \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"group_id": "..."}'

# Archive session
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/archive \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Regenerate Flutter Models
After validating backend:

```bash
# Using openapi-generator (or your existing setup)
openapi-generator generate \
  -i http://localhost:8000/openapi.json \
  -g dart \
  -o app/generated/api

# Commit updated models
git add app/generated/api/
git commit -m "chore: regenerate Flutter models for session access refactor"
```

## Phase 8: Testing & Validation (Next 2-3 Days)

### 8.1 Automated Test Suite

Create comprehensive test files:

**`backend/tests/services/test_session_access_service.py`**
- Test direct access grant/revoke
- Test group linkage workflows
- Test origin tracking
- Test cascade cleanup
- Test idempotent operations

**`backend/tests/api/v1/test_session_access.py`**
- Test all endpoints for authorization
- Test payload validation
- Test error responses
- Test role validation

**`backend/tests/api/v1/test_session_user_state.py`**
- Test archive/unarchive
- Test delete/restore
- Test state query
- Test user isolation

**`backend/tests/core/test_authorization.py`**
- Test require_session_access
- Test require_session_owner
- Test non-leaky 404s
- Test visibility filtering

### 8.2 Load & Performance Testing

Test with realistic data volumes:

```python
# Test with multiple users and groups
# Measure access check performance
# Verify index usage
# Check query optimization

# Load test: 1000 sessions, 100 users, 50 groups
# Expected response time: < 100ms for access queries
```

### 8.3 Data Migration Validation

For each environment:

1. **Backup database**
2. **Run migration**
3. **Verify data integrity**
   - Count SessionAccess records
   - Count SessionAccessOrigin records
   - Verify owner_id populated
   - Verify cascade works
4. **Rollback test** (verify downgrade works)
5. **Re-apply migration**

## Phase 9: Client Refactor (Weeks 2-4)

Follow Workstreams A-C from the original plan:

### 9.1 Workstream A: Navigation Refactor

**A1 - Remove Groups from primary navigation**
- Remove Groups tab from main navigation
- Redirect existing group links to new location
- Update route configuration

**A2 - Unified routines list**
- Create unified routines feed
- Show routines user owns + sessions they have access to
- Add filter chips (owned, collaborating, archived)
- Implement pagination

**A3 - Session creation from routine context**
- Add "Create Session" action in routine detail
- Session name/label optional
- Auto-grant user admin access

**A4 - Access sharing from groups**
- "Share" action on routine detail
- Select groups to grant access to
- Shows current group links
- Ability to remove group access

### 9.2 Workstream B: Casting UI in Routine Detail

**B1 - Data required by routine detail**
- Query session participants
- Query dancer slots
- Query session access (for available users)
- Lazy load as needed

**B2 - Replace `_DancersSection` with `_CastingSection`**
- Redesigned UI for casting
- Show roles (dancer, coach)
- Show available users
- Show already-cast users

**B3 - Participant creation on drag-and-drop**
- Drag user to slot → create participant
- POST /sessions/{id}/participants (atomic)
- Visual feedback during creation
- Error handling for conflicts

**B4 - Participant and slot behavior**
- Update participant role
- Move participant to different slot
- Remove participant (returns to available)
- Slot constraints enforcement

**B5 - Reordering slots**
- Drag to reorder dancer slots
- PUT /routine-dancer-slots/{id}/reorder
- Visual feedback
- Undo capability (optional)

**B6 - Session management sheet**
- Bottom sheet to manage session
- Show access rules
- Show group links
- Manage participants

**B7 - Manage Groups destination**
- Link to Groups section for access management
- Show which groups have access
- Add/remove group access

### 9.3 Workstream C: Removal Flows

**C1 - Owner routine deletion**
- Owner can delete routine
- Deletes all associated sessions
- Confirm dialog with consequences

**C2 - Session archiving for current user**
- Archive/unarchive session
- Restores to archived list section
- Doesn't affect other users

**C3 - Non-owner access removal**
- Revoke own access from session
- Confirms action
- Session removed from user's list

## Phase 10: Integration & Polish (Week 4)

### 10.1 End-to-End Testing
- Create routine → create session → share with group → add participants → archive
- All user roles working correctly
- Permission checks enforced
- Error messages helpful

### 10.2 Performance Optimization
- Cache session access queries
- Lazy load participant lists
- Optimize group member queries
- Monitor API response times

### 10.3 Documentation
- Update API documentation
- Frontend implementation guide
- Migration guide for existing data
- Troubleshooting guide

## Phase 11: Deployment (Week 5)

### 11.1 Staging Validation
1. Deploy to staging environment
2. Run full test suite
3. Smoke test all workflows
4. Performance profiling
5. Security audit

### 11.2 Production Deployment
1. Backup production database
2. Run migration with monitoring
3. Deploy backend code
4. Deploy Flutter app to beta
5. Monitor error rates and performance
6. Gradual rollout to production

### 11.3 Monitoring & Observability
- Track API endpoint latencies
- Monitor database migration completion
- Watch error rates on new endpoints
- Alert on authorization failures
- Log access pattern changes

## Implementation Details for Each Phase

### For Phase 9.1 (Navigation)
Files to modify:
- `lib/config/routes.dart` - Remove groups tab
- `lib/features/routines/presentation/pages/routines_list_page.dart` - New unified feed
- `lib/features/routines/presentation/controllers/routines_controller.dart` - Feed controller

### For Phase 9.2 (Casting)
Files to create/modify:
- `lib/features/routines/presentation/widgets/casting_section.dart` - New
- `lib/features/routines/presentation/widgets/participant_list.dart` - New
- `lib/features/routines/presentation/widgets/session_management_sheet.dart` - New
- Update participant model generation

### For Phase 9.3 (Removal)
Files to modify:
- Dialog implementations
- Confirmation flows
- Error handling

## Known Limitations & Future Work

1. **Participant Role Constraints**
   - Currently: dancer, coach
   - Future: support other roles as migration ready

2. **Access Expiration**
   - Not implemented yet
   - Could add time-limited sharing

3. **Audit Logging**
   - Origins tracked but not logged
   - Could add access change logs

4. **Batch Operations**
   - Single operations only
   - Could optimize bulk access grants

5. **Advanced Permissions**
   - Role-based capabilities not differentiated
   - All admins have same permissions

## Rollback Plan

If issues arise:

1. **Backend Issues**
   ```bash
   alembic downgrade -1
   # Deploy previous code version
   ```

2. **Client Issues**
   ```bash
   # Roll back Flutter release
   # Revert to previous version
   ```

3. **Data Issues**
   ```bash
   # Restore database backup
   # Re-run migration if needed
   ```

## Success Metrics

Track these KPIs during rollout:

- [ ] API endpoint latency < 100ms (p95)
- [ ] Access check success rate > 99.9%
- [ ] Migration completed without data loss
- [ ] No regressions in existing workflows
- [ ] Error rate < 0.1% on new endpoints
- [ ] User satisfaction with new UI
- [ ] Adoption of new sharing features

## Questions to Resolve

Before Phase 9, answer:

1. Should archived sessions be visible in main list or separate section?
2. What's default role for group-linked access? (Currently: admin)
3. Should session deletion be hard-delete or soft-delete?
4. Should participants auto-remove when user loses access?
5. Can multiple users edit same session concurrently?
6. What's the max number of groups per session?
7. Should group role override individual participant role?

## Contacts & Escalation

For questions/issues:
- Backend changes: Backend team
- Frontend implementation: Frontend team
- Database migrations: DevOps team
- Production deployment: Release management

## Timeline Summary

```
Week 1: Testing & Validation
  ├─ Run migration tests
  ├─ API endpoint validation
  ├─ Generate Flutter models
  └─ Prepare release notes

Week 2-3: Client Implementation
  ├─ Navigation refactor (A)
  ├─ Casting UI (B)
  └─ Removal flows (C)

Week 4: Integration
  ├─ End-to-end testing
  ├─ Performance optimization
  └─ Documentation

Week 5: Deployment
  ├─ Staging validation
  ├─ Production deployment
  └─ Monitoring & support
```

## Final Checklist

Before going live:

- [ ] All tests passing
- [ ] Migration verified in staging
- [ ] Flutter models regenerated
- [ ] Client UI implemented
- [ ] Documentation complete
- [ ] Monitoring configured
- [ ] Rollback plan tested
- [ ] Team trained
- [ ] Success metrics defined
- [ ] Launch date scheduled
