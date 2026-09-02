
# Standard imports
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'bing'
setup_keywords['description'] = 'The Bayesian INferences with Gordon coefficents (BING) package' 
setup_keywords['author'] = 'J. Xavier Prochaska, R. Frouin'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/AI-for-Ocean-Science/bing'
setup_keywords['version'] = '0.0.dev0'
# Use README.rst as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['provides'] = [setup_keywords['name']]
# Floor matches retrieve-or-bust's python_requires ('>=3.12'), which bing
# imports unconditionally since the rob_rt integration (PR #27); see
# claude_prompts/RT/rob_rt_prompt_6.md Q8 for the decision trail.
setup_keywords['requires'] = ['Python (>=3.12.0)']
setup_keywords['install_requires'] = [
    'seaborn', 'smart-open[s3]',
    'scikit-learn', 'scikit-image', 'tqdm', 'astropy', 'astropy-healpix',
    'healpy', 'cftime', 'bokeh', 'umap-learn', 'llvmlite', 'boto3',
    'xarray', 'h5netcdf', 'emcee', 'corner',
    'importlib-metadata', 'timm==0.3.2', 'IPython',
    'scikit-learn', 'scikit-image', 'tqdm',
    'pysolar','pytest',
    # robust.rt backend (rob_rt integration, PR #27): retrieve-or-bust is
    # not on PyPI, so a bare name here breaks any plain `pip install .`
    # ("ERROR: No matching distribution found for retrieve-or-bust" --
    # which is exactly how the ReadTheDocs build died). A PEP 508 direct
    # reference tells pip to fetch it from GitHub instead; pip supports
    # this natively in install_requires, and a non-editable install ships
    # the emulator weights because retrieve-or-bust's setup.py declares
    # them in package_data. Pinned to the cdom-rt branch because robust/rt
    # does not exist on that repo's main yet -- same pin as
    # .github/workflows/tests.yml (Q6 in claude_prompts/RT/
    # rob_rt_prompt_6.md); re-point or drop the @cdom-rt once robust/rt
    # merges to main. retrieve-or-bust's own setup.py deliberately
    # excludes the JAX stack from its install_requires (kept in its
    # requirements.txt only) -- so jax/flax/jaxtyping are listed here
    # explicitly rather than assumed transitive. optax is NOT needed:
    # robust only imports it lazily inside its own emulator-training
    # functions, never on the inference path bing uses.
    'retrieve-or-bust @ git+https://github.com/ocean-colour/retrieve-or-bust.git@cdom-rt',
    'jax', 'flax', 'jaxtyping']
setup_keywords['extras_require'] = {
    # Docs build: pip install -e ".[docs]"
    #   keep in sync with docs/requirements.txt (used by ReadTheDocs)
    'docs': ['sphinx>=4.5.0', 'sphinx-rtd-theme>=1.0.0',
             'docutils>=0.18'],
}
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages()
setup_keywords['setup_requires'] = ['pytest-runner']
setup_keywords['tests_require'] = ['pytest']

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
