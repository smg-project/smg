//! What a policy document says, through the parser, the validator and a
//! decision. One program: each module is a file of this folder and holds
//! the cases of its own subject.
//!
//! A file stands here where its cases need no build feature and no process
//! of their own. A case that needs a process, a feature or a program's
//! name of its own stands outside this program.

mod corpus;
mod filter_params;
mod has_room_rule;
mod least_cost;
mod picker_params;
mod schema;
mod scorer_params;
mod signal_terms;
mod signals;
