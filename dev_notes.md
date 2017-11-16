# Guide for developers
## Developing on...
### Windows
Install [Cygwin64 Terminal](https://cygwin.com/install.html) and select all packages.

### Ubuntu
    $ sudo add-apt-repository ppa:fkrull/deadsnakes
    $ sudo apt-get install git python36

### Mac OS X
Download [python](https://www.python.org/ftp/python/3.6.3/python-3.6.3-macosx10.6.pkg) and follow instructions.

## Git set-up
### Basic config
```
    $ git config --global user.name "NAME"
    $ git config --global user.email "EMAIL@EXAMPLE.COM"
    $ git config --global credential.helper 'store --file ~/.git-credential'
```
### Cloning repo
* First, set **pymaat** root path:
    `$ export PYMAAT_PATH=~/pymaat/`
* Then, clone repository:
    `$ git clone https://github.com/hugolamarrephd/pymaat/ $PYMAAT_PATH`
* Enter Github username/password. Remember that if you use two-factor identification you must
generate a permanent **token** from *Settings>Developer settings>Personal access tokens* and use it in lieu of a password.

### Contrib guidelines
#### Basic Workflow
0. Create your feature branch: `git checkout -b your-local-feature-branch`
1. Always run `$ git diff ` before staging (or `$ git diff --staged` when ready to commit) and manually inspect changes.
2. In particular, make sure there are no trailing whitespaces.
3. Craft simple messages (no description) that help **you** identify each commit
4. **Don't push feature branch until ready for pull request**
#### Pull requests to `master`
0. Update repo: `$ git fetch`
1. Be sure to be in feature branch: `$ git checkout your-local-feature-branch`
2. Double check **everything**: `git diff HEAD origin/master`
3. Merge your work: `git rebase -i origin/master`
4. Clean history:
    * Squash related commits using `s`
    * Write meaningful commit name *and* description (separated by line
     break)
    * Try to be as detailed as possible to help **others** review your work
4. Push to Github: `git push`
5. Navigate to your local feature branch:
https://github.com/hugolamarrephd/pymaat/tree/your-local-feature-branch
6. Click *New pull request* and provide relevant information about your new
 feature (with base set to `master`)

## Setting-up virtualenvwrapper
### Installation
```
    $ python3.6 -m pip install virtualenvwrapper
    $ export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
    $ source $(find / -type f -name virtualenvwrapper.sh -print -quit)
    $ mkvirtualenv pymaat
```
### Dependencies
```
    $ workon pymaat
    $ cd $PYMAAT_PATH
    $ pip install -r requirements.txt
```
## Bash set-up (in `~/.bash_profile`)
1. Set **pymaat** root path;
2. Add root to python path;
3. Set-up virtual environment;
4. Source virtualenvwrapper shell script;
5. Work-on pymaat by default (can be commented out);
```
    export PYMAAT_PATH=~/pymaat/
    export PYTHONPATH=$PYTHONPATH:$PYMAAT_PATH
    export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
    venv_path=$(find /usr -type f -name virtualenvwrapper.sh -print -quit)
    if [[ -r venv_path ]]; then
        source venv_path
    else
        echo "WARNING: Can't find virtualenvwrapper.sh"
    fi
    workon pymaat
```
## Tests
* Grant execution permission:
    `$ chmod +x tests/run.py`
* Running tests from command line:
    `$ tests/run.py`

## Text Editor
We strongly encourage to use vim as text editor. In any case, make sure:

* each indentation level is separated by exactly 4 white spaces;
* there is no trailing white spaces or end-of-lines at respectively the end of each line or the end of a file;
* .py files respect [PEP8](https://www.python.org/dev/peps/pep-0008/);

before each commit.

### Vim config
Here is basic `~/.vimrc` configuration to get you started:

    * trailing spaces are highlighted in black;
    * use `:tabe` to create new tabs and `<F7>` and `<F9>` to navigate;
    * use `:vs` to make a vertical split (on wide screen) and `<CTRL-H>` and
    `<CRTL-L>`
    to navigate;
    * press `<F5>` to automatically remove all trailing whitespaces;

```
set nocp
filetype off

set encoding=utf-8

"Tab navigation
nnoremap <F7> :tabp<CR>
nnoremap <F9> :tabn<CR>

nnoremap <C-h> :wincmd h<CR>
nnoremap <C-j> :wincmd j<CR>
nnoremap <C-k> :wincmd k<CR>
nnoremap <C-l> :wincmd l<CR>

"Highligh trailing whitespace in py files

highlight BadWhitespace ctermbg=black guibg=black
au BufRead,BufNewFile * match BadWhitespace /\s\+$/

"Use <F5> to remove all trailing whitespaces
nnoremap <silent> <F5>
\:let _s=@/<Bar>
\:%s/\s\+$//e<Bar>
\:let @/=_s<Bar>
\:nohl<Bar>
\:unlet _s <Bar>
\:%s/\($\n\s*\)\+\%$//e<Bar>
\<CR>

" Basic PEP 8
au BufNewFile,BufRead *
    \ set tabstop=4 |
    \ set softtabstop=4 |
    \ set shiftwidth=4 |
    \ set textwidth=79 |
    \ set expandtab |
    \ set autoindent |
    \ set fileformat=unix |

"Show line numbers
set nu

"Highlight matches
set hls is
```
