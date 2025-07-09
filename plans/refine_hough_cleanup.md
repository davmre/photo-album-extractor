# Refine Hough Cleanup Plan

## Current Context

This document tracks the incremental refactoring of `core/refine_strips_hough.py` - a sophisticated Hough transform-based algorithm for detecting photo edges in scanned images.

### Algorithm Overview
The algorithm works in 5 stages:
1. **Border Strip Extraction**: Creates 4 strips around the photo's perimeter
2. **Edge Detection**: Uses gradient filters to find edge weights
3. **Angle Detection**: Uses FFT to find dominant angles across all strips
4. **Intercept Scoring**: For each angle, finds best edge positions using voting
5. **Hypothesis Generation**: Combines all edge combinations and scores them
6. **Result Selection**: Returns the highest-scoring edge combination

### Test Command
```bash
python3 -m tests.test_refinement_on_real_scans
```

## ✅ Phase 1 Complete (All tasks verified working)

### Completed Tasks:
1. **Extract magic numbers to module-level constants** - Added descriptive constants like `FFT_MAX_RADIUS_FRACTION`, `EDGE_WEIGHT_BLUR_KERNEL_SIZE`, etc.
2. **Split `extract_border_strips()` function** - Extracted:
   - `calculate_strip_boundaries()` - handles aspect ratio expansion logic
   - `create_strip_coordinate_transforms()` - creates coordinate transformation functions
3. **Extract coordinate helpers** - Replaced inline lambdas with named functions where beneficial
4. **Break down `radial_profile_corrected()`** - Extracted:
   - `calculate_angle_range_for_radial_profile()` - angle range calculation with aspect corrections
   - `sample_radial_direction()` - radial sampling logic
5. **Verification** - All changes pass the benchmark test

### Benefits Achieved:
- **Improved readability** - Functions focused on single responsibilities
- **Better maintainability** - Magic numbers centralized and named
- **Enhanced testability** - Smaller functions can be tested independently
- **Preserved functionality** - All tests pass, algorithm behavior unchanged
- **Functional style maintained** - No unnecessary objects, kept to pure functions

## ✅ Phase 2A Complete (Documentation wins achieved)

### Completed Tasks:
1. **Added comprehensive docstrings** - All 26 major functions now have detailed documentation explaining purpose, parameters, and algorithm approach
2. **Improved confusing variable names** - Key improvements:
   - `central_angle_perp` → `central_angle_perpendicular`
   - `min_angle`, `max_angle` → `min_angle_sampling`, `max_angle_sampling`
   - `x_min`, `x_max` → `strip_x_min`, `strip_x_max` (context-specific)
   - `strip_left_x` → `strip_boundary_left` (clearer purpose)
3. **Added inline comments** - 10+ comment blocks explaining complex coordinate transformations and aspect ratio logic
4. **Verification** - All changes pass the benchmark test

### Benefits Achieved:
- **Dramatically improved readability** - Code is now much easier to understand for new developers
- **Better maintainability** - Clear variable names and comprehensive documentation make changes safer
- **Enhanced debuggability** - Inline comments explain the reasoning behind complex transformations
- **Preserved functionality** - Algorithm behavior is completely unchanged

## ✅ Phase 2 Complete (Data Flow Improvements)

### Completed Tasks:
1. **✅ Make `StripData` construction cleaner** - Initialized all fields in `__init__` with proper type hints (| None) and added runtime assertions where needed
2. **✅ Extract `score_intercepts_for_strip()` side effects** - Modified function to return `(intercept_scores, intercept_bins)` instead of mutating strip fields
3. **✅ Simplify `enumerate_hypotheses()`** - Extracted `generate_candidate_index_combinations()` helper function for the deeply nested 4-level loop logic
4. **✅ Clean up `get_candidate_edges()`** - Split into separate functions:
   - `boost_boundary_edge_scores()` - Handles image boundary boosting
   - `extract_candidate_peaks()` - Extracts prominent peaks from intercept scores
   - `convert_intercepts_to_edges()` - Converts intercepts to edge coordinates
   - `get_candidate_edges()` - Orchestrates the overall process

### Benefits Achieved:
- **Eliminated side effects** - Functions now return values instead of mutating global state
- **Improved data flow clarity** - Algorithm stages are more explicit and predictable
- **Better function organization** - Each function has a single, clear responsibility
- **Enhanced maintainability** - Easier to modify individual components without affecting others
- **Preserved functionality** - All tests pass, algorithm behavior completely unchanged

## 📋 Remaining Cleanup Plan

### Phase 3: Simplify Complex Logic
1. **Break down `intercept_of_line_touching_image_edge()`** - This 50-line function has complex coordinate handling that could be extracted
2. **Simplify `find_best_overall_angles()`** - Extract the angle scoring combination logic
3. **Clean up `get_candidate_edges()`** - Split into separate functions for:
   - Scoring intercepts per strip
   - Boosting boundary edges  
   - Finding peak candidates
   - Converting to image coordinates

### Phase 4: Advanced Improvements (Optional)
1. **Extract debug plotting** - Separate debug code from main logic (called conditionally)
2. **Add comprehensive type hints** - Complete type annotation coverage
3. **Consider data structures** - Evaluate if the nested dictionaries in `StripData` could be simplified

### Verification Strategy for Each Phase:
- Run `python3 -m tests.test_refinement_on_real_scans` after each step
- Compare debug output images to ensure visual results are identical
- Add simple unit tests for extracted pure functions (optional)

## 🎯 Expected Final Benefits:
- **Single responsibility** - Each function has one clear purpose
- **Testability** - Individual components can be tested in isolation  
- **Readability** - Clear flow and better names throughout
- **Maintainability** - Easier to modify individual components without affecting others
- **Debugging** - Debug code separated from main logic
- **Documentation** - Comprehensive understanding of algorithm steps

## 📝 Implementation Notes:
- Keep changes incremental and verifiable
- Maintain functional style - avoid unnecessary object creation  
- Each change should be small enough to easily verify against the benchmark
- Preserve exact algorithm behavior - no performance or accuracy changes
- Focus on code organization and clarity, not algorithmic improvements

## 🔍 Key Insights from Phases 1 & 2A:
- **Function extraction works well** - The code responded nicely to breaking apart large functions
- **Constants cleanup was high-impact** - Centralizing magic numbers made the code much more readable
- **Documentation has massive impact** - Adding docstrings and improving variable names dramatically improved readability
- **Coordinate transformations are the biggest complexity** - Multiple coordinate systems (image, strip, normalized) cause confusion
- **Side effects are the main data flow issue** - Functions that mutate `StripData` objects make the algorithm harder to follow
- **The algorithm is well-structured** - The high-level flow is logical, it's the implementation details that need cleaning

## 🎯 Updated Priorities (Based on Results):
1. **✅ Documentation completed** - This had massive impact on readability and should be the model for future work
2. **Data flow improvements** - Eliminating side effects will make the algorithm much clearer  
3. **Complex functions** - Some functions like `intercept_of_line_touching_image_edge` really need breaking down
4. **Debug code separation** - This is more of a nice-to-have than essential

## 🔧 Recommended Next Steps:
Based on the success of Phase 2A, the next logical step is **Phase 2** (Data Flow improvements), which will:
- Build on the improved readability from documentation
- Make the algorithm easier to follow by eliminating side effects
- Prepare the code for easier testing and debugging
- Continue the pattern of small, verifiable changes