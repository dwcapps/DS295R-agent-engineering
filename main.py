def get_response(prompt: str) -> str:
    return prompt

def main():
    while True:
        try:
            prompt = input("Input: ")
            if prompt == "exit": break
            response = get_response(prompt)
        except EOFError:
            break
        print(response)

if __name__ == "__main__":
    main()