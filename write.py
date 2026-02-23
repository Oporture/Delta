import os
import sys
import re
from colorama import init, Fore, Style
from pynput import keyboard

# Initialize colorama
init(autoreset=True)

EXAMPLES_DIR = os.path.join(os.getcwd(), "examples")

class WriterState:
    def __init__(self):
        self.role = "User"
        self.messages = []
        self.should_switch = False
        self.should_save = False
        self.running = True

    def get_role_color(self):
        return Fore.CYAN if self.role == "User" else Fore.YELLOW

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
kb_controller = keyboard.Controller()

def on_press(key):
    try:
        is_np1 = False
        is_np2 = False
        
        # pynput handles numpad keys as Key.kp_X
        if key == keyboard.Key.kp_1:
            is_np1 = True
        elif key == keyboard.Key.kp_2:
            is_np2 = True
        
        # On some keyboards/OSs, they might come as characters with specific VKs
        # or just as the characters themselves.
        # But for 'numpad', kp_1 is the standard.
        
        if is_np1:
            state.should_switch = True
            kb_controller.press(keyboard.Key.enter)
            kb_controller.release(keyboard.Key.enter)
        elif is_np2:
            state.should_save = True
            kb_controller.press(keyboard.Key.enter)
            kb_controller.release(keyboard.Key.enter)

    except Exception:
        pass

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}--- delta-dataset Writer ---")
    print(f"{Fore.WHITE}Type message and hit Enter to add.")
    print(f"{Fore.WHITE}Numpad 1: Switch Role | Numpad 2: Save and Exit")
    print("-" * 40)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while state.running:
            color = state.get_role_color()
            prompt = f"{color}{Style.BRIGHT}{state.role}:{Style.RESET_ALL} "
            
            sys.stdout.write(prompt)
            sys.stdout.flush()
            
            user_input = sys.stdin.readline().strip()

            # Add message if there's any text
            if user_input:
                # Format as requested (implied by md): **Role**: msg
                state.messages.append(f"**{state.role}**: {user_input}")

            if state.should_save:
                if state.messages:
                    filename = state.get_next_filename()
                    with open(filename, "w") as f:
                        f.write("\n\n".join(state.messages))
                    print(f"\n{Fore.GREEN}✔ Saved {len(state.messages)} messages to {filename}")
                else:
                    print(f"\n{Fore.RED}✘ Nothing to save.")
                state.running = False
                break

            if state.should_switch:
                state.role = "Assistant" if state.role == "User" else "User"
                state.should_switch = False
                print(f"{Fore.MAGENTA}Role switched to {state.role}")
                continue
                
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exiting...")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
