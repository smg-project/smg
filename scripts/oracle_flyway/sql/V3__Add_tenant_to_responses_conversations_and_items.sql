-- V3: Add the TENANT column to RESPONSES, CONVERSATIONS and CONVERSATION_ITEMS
-- (the gateway's core history migration v4: the tenant a record is scoped to;
-- rows from before the column stay untenanted).

ALTER TABLE RESPONSES ADD (TENANT VARCHAR2(255));
ALTER TABLE CONVERSATIONS ADD (TENANT VARCHAR2(255));
ALTER TABLE CONVERSATION_ITEMS ADD (TENANT VARCHAR2(255));
