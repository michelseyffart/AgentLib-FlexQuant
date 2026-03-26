# Changelog

## 0.2.0

### New Features
- Added support for distinguishing feed-in and consumption electricity tariffs in the flexibility indicator (new `price_variable_feed_in` config/input in `FlexibilityIndicator`, `feed_in_price_series` field in `FlexibilityKPIs` and `FlexibilityData`)
- Added multiprocessing support for loading simulation results, improving performance on large result sets
- Added support for custom objectives in shadow MPCs
- Added baseline power prediction as an input to shadow MPCs
- Included CasADi-based simulator as an optimization backend option
- Added support for using a control variable as the flexibility boundary
- Added CI test for the `OneRoom_CIA` example
- Added coverage report and badge
- Added repository logo

### Changes
- Improved MPC callback structure: the Baseline MPC now sends its full knowledge (states, inputs, and controls) to shadow MPCs, ensuring they operate on consistent information; shadows wait for all variables before executing their step
- Adjusted handling of custom inputs in MPC modules
- Adapted results path function and example scripts
- Updated package requirements and dependencies

### Bug Fixes
- Fixed simulation results file path resolution
- Fixed alias handling in MPC modules
- Fixed CIA (Combined Integer Approach) backend
- Fixed cost calculation in flexibility KPIs
- Fixed path input handling
- Fixed `agentlib_mpc` import
- Removed erroneous error raise in flexibility indicator
- Improved pylint score and code quality

## 0.1.0

- Added first version of code
- Added CHANGELOG
- Added CI integration
