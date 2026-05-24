 # Saving results from BING fits to files

## Goals

Capture all of the main outputs of a BING fit and the inputs required to reproduce the fit.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the repository
- Add inline comments to explain the effort
- Use methods, not classes
- Place import statements at the top of the file.
- Use numpy.savez() to save arrays 
- Use JSON files for inputs and simple stats

## Testing

If you need to run python, use the "ocean14" environment in conda.

## Docs

Examine the files in the docs/ directory and update the docs to reflect the new changes.  In particular:

## Tests

1. Update the tests in the bing/tests/test_chl_fl.py file to include tests for the new functionality.

2. Check that the Chl tests in test_l23_fitting.py are passing.

## Prompts

1.  Generate/update the docs as described in the Docs section above.

2. Re-read this doc.  Execute the first item in the Tests section above.

3. Re-read this doc.  Execute the 2nd item in the Tests section above.