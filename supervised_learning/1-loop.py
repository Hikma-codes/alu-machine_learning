#!/usr/bin/env python3
'''
Script that takes in user input with the prompt 'Q:' and
prints 'A:' as the response.
'''


if __name__ == "__main__":
    while (1):
        user_input = input("Q: ")
        user_input = user_input.lower()
        if user_input == 'exit' or user_input == 'quit' \
           or user_input == 'goodbye' or user_input == 'bye':
            print("A: Goodbye")
            break
        print("A:")
