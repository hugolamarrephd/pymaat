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
### Basic confif
    $ git config --global user.name "NAME"
    $ git config --global user.email "EMAIL@EXAMPLE.COM"
    $ git config --global credential.helper 'store --file ~.git-credential'
### Cloning repo
* First, select root path for project:
    export PYMAAT_PATH=~/pymaat/
* Then, clone repository:
    $ git clone https://github.com/hugolamarrephd/pymaat/ $PYMAAT_PATH
* Enter Github username/password. Remember that if you use two-factor identification you must 
generate a permanent **token** from *Settings>Developer settings>Personal access tokens* and use it in lieu of a password.

## Setting-up virtualenvwrapper
### Installation
    $ python3.6 -m pip install virtualenvwrapper
    export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
    $ source $(find / -type f -name virtualenvwrapper.sh -print -quit)
    $ mkvirtualenv pymaat

### Dependencies
    $ workon pymaat
    $ cd $PYMAAT_PATH
    $ pip install -r requirements.txt 

## Bash set-up (in '~/.bash\_profile')
First, set root of project as environment variable for convenience:
    export PYMAAT_PATH=~/pymaat/
Add root to python path:
    export PYTHONPATH=$PYTHONPATH:$PYMAAT_PATH
Set-up virtual environment
    export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
Source virtualenvwrapper shell script:
    source $(find / -type f -name virtualenvwrapper.sh -print -quit)
Work-on pymaat by default (can be commented out)
    workon pymaat  
 
## Tests
Grant execution permission:
    $ chmod +x tests/run.py	
Running tests from command line:
    $ tests/run.py

## Text Editor
We strongly encourage vim as text editor. In any case, make sure:
* each indentation level is separated by exactly 4 white spaces;
* there is no trailing white spaces or end-of-lines at respectively to end of each line and the end of a file
before each commit.

