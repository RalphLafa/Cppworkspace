#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

bool isPalinedrome(char* s);

int main() {
    char* result = (char*)malloc(sizeof(char)* 5);
    char str[5] = "abcba";
    for(int i = 0; i < strlen(str); i++) 
        result[i] = str[i];


    if(isPalinedrome(result))
        printf("good");

    return 0;

}

bool isPalinedrome(char* s) {
    int bias_left  = 0;
    int bias_right = 1;

    int str_len = strlen(s);
    for (int i = 0; i < str_len; i++) {
        while (!isalnum(s[i + bias_left])) {
            bias_left++;
            if (i + bias_left == str_len)
                return true;
        }
        while ( !isalnum(s[str_len - i - bias_right])){
            bias_right++;
        }
        if (i + bias_left >= str_len - i - bias_right)
            break;
        if (tolower(s[i+bias_left]) != tolower(s[str_len - i - bias_right]))
            return false;

    }
    return true;
}