/*
Firstly, you need to install mingw64 
and add it to your environment variable path

**How to run your C code
 
terminal : powershell, cmd

1. C:\Cppworkspace>gcc main.c
-> 'a.exe' file created
2. C:\Cppworkspace>.\a.exe
-> Run your main.c file

OR

C:\Cppworkspace>gcc main.c -o main.exe
-> 'main.exe' created
C:\Cppworkspace>.\main.exe


*/


#include <stdio.h>

int main() {
    printf("Hello world!");

    return 0;
}