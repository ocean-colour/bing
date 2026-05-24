 # Chl-a Fluorescence 

## Goals

Adds Chl Fl to the bing.rt module.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the repository
- Add inline comments to explain the effort
- Use methods, not classes
- Place import statements at the top of the file.

## Docs

Examine the files in the docs/ directory and update the docs to reflect the new changes.  In particular:

- Make sure the docs/chlorophyll_fluorescence.rst file is up to date.
- Add a new section to the docs/index.rst file to include that file
- Update the docs/radiative_transfer.rst file, as needed
- Note the new dependency on the correct_atmosphere repository, which is located at https://github.com/ocean-colour/correct-atmosphere
- Update the docs/models.rst file, as needed
- Update the docs/parameters.rst file, as needed
- Add docs on how to instantiate and use the rt_dict_from_p() function


## Tests

1. Update the tests in the bing/tests/test_chl_fl.py file to include tests for the new functionality.

## Prompts

1.  Generate/update the docs as described in the Docs section above.

2. Re-read this doc.  Execute the first item in the Tests section above.