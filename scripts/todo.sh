rm -f TODO.txt
grep -rnw . -e 'TODO' > TODO.txt
