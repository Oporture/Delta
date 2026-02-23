import os
import sys
import re
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

EXAMPLES_DIR = os.path.join(os.getcwd(), "examples")

class WriterState:
    def __init__(self):
        self.role = "user"
        self.messages = []
        self.should_switch = False
        self.should_save = False
        self.running = True

    def get_role_color(self):
        return Fore.CYAN if self.role == "user" else Fore.YELLOW

    def get_next_filename(self):
        if not os.path.exists(EXAMPLES_DIR):
            os.makedirs(EXAMPLES_DIR)
        
        files = os.listdir(EXAMPLES_DIR)
        numbers = []
        for f in files:
            if f.endswith(".md"):
                match = re.match(r"(\d+)\.md", f)
                if match:
                    numbers.append(int(match.group(1)))
        
        next_num = max(numbers) + 1 if numbers else 1
        return os.path.join(EXAMPLES_DIR, f"{next_num}.md")

state = WriterState()

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}--- delta-dataset Writer ---")
    print(f"{Fore.WHITE}Type message. Hit Enter to Switch Role.")
    print(f"{Fore.WHITE}Hit Enter on an empty prompt to Save and Exit.")
    print("-" * 40)

    try:
        while state.running:
            color = state.get_role_color()
            role_disp = state.role.capitalize()
            prompt = f"{color}{Style.BRIGHT}{role_disp}:{Style.RESET_ALL} "
            
            sys.stdout.write(prompt)
            sys.stdout.flush()
            
            # Read a single line (Enter key)
            user_input = sys.stdin.readline().strip()

            # If text was entered, add message and switch role
            if user_input:
                state.messages.append(f"<{state.role}>{user_input}</{state.role}>")
                state.role = "assistant" if state.role == "user" else "user"
                # Indication of switch is less necessary if prompt changes immediately, 
                # but following user request for Enter to switch role.
            else:
                # Entered empty line immediately -> Save and Exit
                if state.messages:
                    filename = state.get_next_filename()
                    with open(filename, "w") as f:
                        f.write(" ".join(state.messages))
                    print(f"\n{Fore.GREEN}✔ Saved {len(state.messages)} messages to {filename}")
                else:
                    print(f"\n{Fore.RED}✘ Nothing to save.")
                state.running = False
                break
                
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exiting...")

if __name__ == "__main__":
    main()
