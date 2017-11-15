# Guide for developers
## Developing on... 
### Windows
Install [Cygwin64 Terminal](https://cygwin.com/install.html) and select all packages.

### Ubuntu
    $ sudo add-apt-repository ppa:fkrull/deadsnakes
    $ sudo apt-get install git python36

### Mac OS X
Download [installer](https://www.python.org/ftp/python/3.6.3/python-3.6.3-macosx10.6.pkg) and follow instructions.

## Git set-up
### Basic config
```
    $ git config --global user.name "NAME"
    $ git config --global user.email "EMAIL@EXAMPLE.COM"
    $ git config --global credential.helper 'store --file ~.git-credential'
```
### Cloning repo
* First, set **pymaat** root path:
    `$ export PYMAAT_PATH=~/pymaat/`
* Then, clone repository:
    `$ git clone https://github.com/hugolamarrephd/pymaat/ $PYMAAT_PATH`
* Enter Github username/password. Remember that if you use two-factor identification you must 
generate a permanent **token** from *Settings>Developer settings>Personal access tokens* and use it in lieu of a password.

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
Line-by-line description:
1. Set **pymaat** root path;
2. Add root to python path;
3. Set-up virtual environment;
4. Source virtualenvwrapper shell script;
5. Work-on pymaat by default (can be commented out);
```
    export PYMAAT_PATH=~/pymaat/
    export PYTHONPATH=$PYTHONPATH:$PYMAAT_PATH
    export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
    source $(find / -type f -name virtualenvwrapper.sh -print -quit)
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

before each commit. Always run `$ git diff ` before staging (or `$ git diff --staged` when ready to commit) 
and manually inspect changes. Trailing spaces should be highlighted in red by default. 

