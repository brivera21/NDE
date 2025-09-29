"""
DIGIT COMPARISON EXPERIMENT - Based on Moyer & Landauer (1967)

This program recreates the classic psychology experiment that discovered the "distance effect" 
in numerical cognition - people are faster at comparing numbers that are farther apart.

WHAT THIS PROGRAM DOES:
- Shows participants two digits side by side
- Measures how quickly they identify which digit is larger
- Analyzes the data to reveal psychological patterns in number processing
- Saves results for scientific analysis

HARDWARE REQUIREMENTS:
- Computer with keyboard
- Monitor (will go fullscreen)
- Python with required packages (pygame, pandas, numpy, matplotlib)

EXPERIMENT STRUCTURE:
1. Instructions and setup
2. 5 practice trials (easy comparisons)
3. 45 experimental trials (all possible digit pairs 0-9 where one > other)
4. Data analysis and visualization
"""

# IMPORT SECTION - These are the tools/libraries our program needs
import pygame          # For creating the visual experiment window and handling user input
import random          # For randomizing trial order (important for unbiased results)
import time            # For measuring reaction times with millisecond precision
import pandas as pd    # For organizing and saving data in spreadsheet format
import numpy as np     # For mathematical calculations and statistical analysis
import matplotlib.pyplot as plt  # For creating graphs and charts of results
from itertools import combinations  # For generating all possible digit pairs
import sys            # For system-specific operations (like detecting Windows/Mac)
import os             # For operating system commands (like bringing window to front)

class DigitComparisonExperiment:
    """
    MAIN EXPERIMENT CLASS
    
    This is the "brain" of our experiment. It contains all the functions needed to:
    - Set up the visual display
    - Show instructions to participants  
    - Run individual trials
    - Record data
    - Analyze results
    
    Think of this as the experiment protocol that a researcher would follow,
    but written in computer code.
    """
    
    def __init__(self, fullscreen=True):
        """
        INITIALIZATION - Setting up the experiment
        
        This function runs automatically when the experiment starts.
        It's like setting up the lab before participants arrive.
        """
        
        # START THE GRAPHICS SYSTEM
        pygame.init()  # This "wakes up" pygame so we can create windows and show images
        
        # BRING WINDOW TO FRONT (so participants can't miss it)
        # Different computer types need different commands:
        if sys.platform.startswith('win'):
            # Windows computers
            try:
                import win32gui
                import win32con
                import win32process
                # Set this program as high priority so timing is accurate
                win32process.SetPriorityClass(win32process.GetCurrentProcess(), win32process.HIGH_PRIORITY_CLASS)
            except ImportError:
                pass  # If the special Windows tools aren't available, just continue
        elif sys.platform == 'darwin':
            # Mac computers  
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        
        # CREATE THE EXPERIMENT WINDOW
        if fullscreen:
            # Take over the entire screen (like a movie theater)
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.NOFRAME)
        else:
            # Create a regular window
            self.screen = pygame.display.set_mode((1024, 768))
        
        # GET SCREEN DIMENSIONS (so we know where to put things)
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Digit Comparison Experiment")  # Window title
        
        # FORCE WINDOW TO APPEAR IN FRONT
        pygame.display.flip()  # Update the display
        pygame.event.pump()    # Process any pending events
        
        # MORE WINDOW-FOCUSING CODE (for Windows computers)
        if sys.platform.startswith('win'):
            try:
                import win32gui
                import win32con
                hwnd = pygame.display.get_wm_info()["window"]
                # These are technical commands to force the window to the front
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, 
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                win32gui.SetForegroundWindow(hwnd)
            except ImportError:
                pass  # If special Windows tools aren't available, just continue
        
        # DEFINE COLORS (using RGB values - Red, Green, Blue from 0-255)
        self.BLACK = (0, 0, 0)      # Pure black
        self.WHITE = (255, 255, 255) # Pure white  
        self.GRAY = (128, 128, 128)  # Medium gray
        self.GREEN = (0, 255, 0)     # Pure green
        self.RED = (255, 0, 0)       # Pure red
        
        # SET UP FONTS (like choosing font and size in Microsoft Word)
        self.font_large = pygame.font.Font(None, 96)   # Big font for the digits (96 point - increased from 72)
        self.font_medium = pygame.font.Font(None, 36)  # Medium font for titles (36 point)
        self.font_small = pygame.font.Font(None, 24)   # Small font for instructions (24 point)
        
        # EXPERIMENT SETUP VARIABLES
        self.digits = list(range(0, 10))  # Create list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.trial_data = []              # Empty list to store results from each trial
        self.current_trial = 0            # Keep track of which trial we're on
        
        # CREATE ALL POSSIBLE DIGIT PAIRS (where one digit is larger than the other)
        self.digit_pairs = []
        for larger in self.digits:        # For each possible larger digit (0-9)
            for smaller in self.digits:   # For each possible smaller digit (0-9)
                if larger > smaller:      # Only keep pairs where larger > smaller
                    self.digit_pairs.append((larger, smaller))  # Add this pair to our list
        
        # EXAMPLE: This creates pairs like (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), etc.
        # Total: 45 pairs (since we exclude equal digits and duplicates)
        
        # CREATE PRACTICE TRIALS (easy comparisons so people can learn the task)
        self.practice_pairs = [(9, 1), (8, 2), (7, 1), (9, 3), (8, 1)]
        # These are all "easy" comparisons with big differences
        
        # RANDOMIZE THE ORDER (very important for scientific validity!)
        random.shuffle(self.digit_pairs)
        # This mixes up the trials so they don't happen in predictable order
        
        # TIMING VARIABLES (for measuring reaction times)
        self.fixation_duration = 500     # How long to show the + symbol (500 milliseconds)
        self.stimulus_start_time = None  # Will store when digits first appeared
        self.waiting_for_response = False # Flag to track if we're waiting for participant to respond
    
    def show_instructions(self):
        """
        SHOW INITIAL INSTRUCTIONS
        
        This displays the main instructions that explain the task to participants.
        Like reading the consent form and instructions before starting an experiment.
        """
        
        # CREATE THE INSTRUCTION TEXT
        instructions = [
            "Digit Comparison Experiment",                    # Title
            "",                                              # Blank line
            "You will see two digits on the screen.",       # Explain what they'll see
            "Press the F key if the LEFT digit is larger.", # Left response key
            "Press the J key if the RIGHT digit is larger.",# Right response key  
            "",                                              # Blank line
            "Respond as quickly and accurately as possible.", # Speed vs accuracy instruction
            "",                                              # Blank line
            "You will start with 5 practice trials,",       # Practice info
            f"followed by {len(self.digit_pairs)} experimental trials.", # Real experiment info
            "",                                              # Blank line
            "Press SPACE to begin, or ESC to quit."         # How to start/quit
        ]
        
        # DISPLAY THE INSTRUCTIONS
        self.screen.fill(self.WHITE)  # Fill screen with white background
        y_offset = self.height // 2 - len(instructions) * 15  # Calculate starting height
        
        # DRAW EACH LINE OF INSTRUCTIONS
        for line in instructions:
            if line == instructions[0]:  # If this is the title line
                text = self.font_medium.render(line, True, self.BLACK)  # Use medium font
            else:
                text = self.font_small.render(line, True, self.BLACK)   # Use small font
            
            # CENTER THE TEXT HORIZONTALLY
            text_rect = text.get_rect(center=(self.width // 2, y_offset))
            self.screen.blit(text, text_rect)  # Actually draw the text
            y_offset += 30  # Move down for next line
        
        pygame.display.flip()  # Update the screen to show the text
        
        # WAIT FOR PARTICIPANT TO PRESS SPACE OR ESCAPE
        waiting = True
        while waiting:
            for event in pygame.event.get():  # Check for key presses
                if event.type == pygame.QUIT:  # If they closed the window
                    return False
                elif event.type == pygame.KEYDOWN:  # If they pressed a key
                    if event.key == pygame.K_SPACE:    # Space = start experiment
                        return True
                    elif event.key == pygame.K_ESCAPE: # Escape = quit
                        return False
        return False
    
    def show_fixation(self):
        """
        SHOW FIXATION CROSS
        
        The + symbol that appears before each trial.
        This helps participants focus their eyes on the center of the screen
        and provides a consistent starting point for each trial.
        """
        
        self.screen.fill(self.WHITE)  # Clear screen with white background
        
        # DRAW A + SYMBOL IN THE CENTER
        cross_size = 20                           # How big the + should be
        center_x, center_y = self.width // 2, self.height // 2  # Find screen center
        
        # Draw horizontal line of the +
        pygame.draw.line(self.screen, self.BLACK, 
                        (center_x - cross_size, center_y), 
                        (center_x + cross_size, center_y), 3)
        # Draw vertical line of the +
        pygame.draw.line(self.screen, self.BLACK, 
                        (center_x, center_y - cross_size), 
                        (center_x, center_y + cross_size), 3)
        
        pygame.display.flip()  # Show the + on screen
        pygame.time.wait(self.fixation_duration)  # Wait 500ms before continuing
    
    def show_digit_pair(self, left_digit, right_digit):
        """
        SHOW THE TWO DIGITS AND START TIMING
        
        This is the main part of each trial - showing the two numbers
        that the participant needs to compare.
        """
        
        self.screen.fill(self.WHITE)  # Clear screen with white background
        
        # CONVERT NUMBERS TO TEXT AND RENDER THEM
        left_text = self.font_large.render(str(left_digit), True, self.BLACK)   # Left digit
        right_text = self.font_large.render(str(right_digit), True, self.BLACK) # Right digit
        
        # CALCULATE WHERE TO PUT THE DIGITS (farther apart than before)
        left_pos = (self.width // 2 - 150, self.height // 2)   # Left of center (150px from center, increased from 100)
        right_pos = (self.width // 2 + 150, self.height // 2)  # Right of center (150px from center, increased from 100)
        
        # CENTER EACH DIGIT IN ITS POSITION
        left_rect = left_text.get_rect(center=left_pos)
        right_rect = right_text.get_rect(center=right_pos)
        
        # ACTUALLY DRAW THE DIGITS ON SCREEN
        self.screen.blit(left_text, left_rect)
        self.screen.blit(right_text, right_rect)
        
        pygame.display.flip()  # Show the digits
        
        # START THE REACTION TIME TIMER
        self.stimulus_start_time = time.time() * 1000  # Get current time in milliseconds
        self.waiting_for_response = True               # Set flag that we're waiting for response
    
    def show_practice_instructions(self):
        """
        SHOW PRACTICE TRIAL INSTRUCTIONS
        
        Brief reminder about the keys before practice trials begin.
        """
        
        self.screen.fill(self.WHITE)  # Clear screen
        
        instructions = [
            "Practice Trials",                        # Title
            "",                                      # Blank line
            "Let's start with some practice.",       # Explanation
            "",                                      # Blank line
            "Remember:",                             # Reminder header
            "F = Left digit is larger",              # Left key
            "J = Right digit is larger",             # Right key
            "",                                      # Blank line
            "Press SPACE when ready to start practice." # How to continue
        ]
        
        # DISPLAY THE INSTRUCTIONS (same method as main instructions)
        y_offset = self.height // 2 - len(instructions) * 15
        
        for line in instructions:
            if line == instructions[0]:  # Title
                text = self.font_medium.render(line, True, self.BLACK)
            else:
                text = self.font_small.render(line, True, self.BLACK)
            
            text_rect = text.get_rect(center=(self.width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30
        
        pygame.display.flip()
        
        # WAIT FOR SPACE KEY
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
        return False
    
    def show_main_experiment_instructions(self):
        """
        TRANSITION TO MAIN EXPERIMENT
        
        Brief instructions shown after practice, before the real experiment begins.
        """
        
        self.screen.fill(self.WHITE)
        
        instructions = [
            "Great! Now for the main experiment.",     # Encouragement
            "",                                       # Blank line
            "You'll see the same type of trials,",    # Explanation
            "but there will be more of them.",        # More trials coming
            "",                                       # Blank line
            "Continue to respond as quickly",         # Speed reminder
            "and accurately as possible.",            # Accuracy reminder
            "",                                       # Blank line
            "Press SPACE when ready to begin."        # How to start
        ]
        
        # DISPLAY INSTRUCTIONS (same method as before)
        y_offset = self.height // 2 - len(instructions) * 15
        
        for line in instructions:
            text = self.font_small.render(line, True, self.BLACK)
            text_rect = text.get_rect(center=(self.width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30
        
        pygame.display.flip()
        
        # WAIT FOR SPACE KEY
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
        return False
    
    def show_brief_pause(self):
        """
        BRIEF PAUSE BETWEEN TRIALS
        
        Shows a blank screen for 200ms between trials.
        This gives participants a moment to reset before the next trial.
        """
        self.screen.fill(self.WHITE)  # Show blank white screen
        pygame.display.flip()
        pygame.time.wait(200)  # Wait 200 milliseconds
    
    def run_trial(self, trial_num, larger_digit, smaller_digit, is_practice=False):
        """
        RUN A SINGLE TRIAL
        
        This is the heart of the experiment - it runs one complete trial from start to finish.
        
        PARAMETERS:
        - trial_num: Which trial number this is (for record keeping)
        - larger_digit: The digit that is mathematically larger
        - smaller_digit: The digit that is mathematically smaller  
        - is_practice: True for practice trials, False for real trials
        """
        
        # RANDOMLY DECIDE WHICH DIGIT GOES ON WHICH SIDE
        # This is important so participants can't just memorize "left is always bigger"
        if random.choice([True, False]):  # 50% chance
            left_digit, right_digit = larger_digit, smaller_digit  # Larger on left
            correct_response = pygame.K_f  # F key is correct (left digit larger)
        else:
            left_digit, right_digit = smaller_digit, larger_digit  # Larger on right
            correct_response = pygame.K_j  # J key is correct (right digit larger)
        
        # RUN THE TRIAL SEQUENCE
        self.show_fixation()                        # 1. Show + symbol (500ms)
        self.show_digit_pair(left_digit, right_digit)  # 2. Show digits and start timer
        
        # WAIT FOR PARTICIPANT RESPONSE
        response_time = None
        correct = False
        
        while self.waiting_for_response:
            for event in pygame.event.get():  # Check for key presses
                if event.type == pygame.QUIT:  # Window closed
                    return False
                elif event.type == pygame.KEYDOWN:  # Key pressed
                    if event.key in [pygame.K_f, pygame.K_j]:  # F or J pressed
                        # CALCULATE REACTION TIME
                        response_time = time.time() * 1000 - self.stimulus_start_time
                        # CHECK IF RESPONSE WAS CORRECT
                        correct = (event.key == correct_response)
                        self.waiting_for_response = False  # Stop waiting
                    elif event.key == pygame.K_ESCAPE:  # Escape = quit
                        return False
        
        # BRIEF PAUSE BEFORE NEXT TRIAL
        self.show_brief_pause()
        
        # SAVE THE DATA (only for real experimental trials, not practice)
        if not is_practice:
            trial_data = {
                'trial': trial_num,                    # Trial number
                'larger_digit': larger_digit,          # Which digit was mathematically larger
                'smaller_digit': smaller_digit,        # Which digit was mathematically smaller
                'difference': larger_digit - smaller_digit,  # Numerical difference
                'left_digit': left_digit,              # What digit was shown on left
                'right_digit': right_digit,            # What digit was shown on right
                'correct_response': 'left' if correct_response == pygame.K_f else 'right',  # Which side was correct
                'reaction_time': response_time,         # How long they took (milliseconds)
                'correct': correct                     # Whether they got it right
            }
            
            self.trial_data.append(trial_data)  # Add this trial's data to our list
        
        return True  # Trial completed successfully
    
    def run_experiment(self):
        """
        RUN THE COMPLETE EXPERIMENT
        
        This is the "main controller" that runs everything in the right order:
        instructions → practice → main experiment → completion
        """
        
        # SHOW INITIAL INSTRUCTIONS
        if not self.show_instructions():
            return None  # User quit before starting
            
        # RUN PRACTICE TRIALS
        if not self.show_practice_instructions():
            return None  # User quit before practice
            
        print("Running practice trials...")  # Message for researcher
        for i, (larger, smaller) in enumerate(self.practice_pairs):
            if not self.run_trial(i + 1, larger, smaller, is_practice=True):
                return None  # User quit during practice
        
        # TRANSITION TO MAIN EXPERIMENT
        if not self.show_main_experiment_instructions():
            return None  # User quit before main experiment
        
        # RUN MAIN EXPERIMENT
        total_trials = len(self.digit_pairs)
        print(f"Running main experiment ({total_trials} trials)...")  # Message for researcher
        
        for i, (larger, smaller) in enumerate(self.digit_pairs):
            if not self.run_trial(i + 1, larger, smaller, is_practice=False):
                break  # User quit during experiment
        
        # SHOW COMPLETION MESSAGE
        self.screen.fill(self.WHITE)
        text = self.font_medium.render("Experiment Complete!", True, self.BLACK)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)
        
        save_text = "Data saved to 'experiment_results.csv'"
        save_surface = self.font_small.render(save_text, True, self.BLACK)
        save_rect = save_surface.get_rect(center=(self.width // 2, self.height // 2 + 40))
        self.screen.blit(save_surface, save_rect)
        
        pygame.display.flip()
        pygame.time.wait(3000)  # Show completion message for 3 seconds
        
        # RETURN THE DATA AS A SPREADSHEET-LIKE TABLE
        return pd.DataFrame(self.trial_data)

def analyze_results(df):
    """
    ANALYZE THE EXPERIMENTAL RESULTS
    
    This function takes the data from the experiment and analyzes it
    to find the psychological patterns that Moyer & Landauer discovered.
    
    WHAT IT DOES:
    - Calculates accuracy and average reaction times
    - Tests for the "distance effect" (faster responses for larger differences)
    - Computes correlations like in the original 1967 study
    - Creates graphs showing the results
    """
    
    # CHECK IF WE HAVE DATA TO ANALYZE
    if df is None or df.empty:
        print("No data to analyze")
        return
    
    # FILTER TO ONLY CORRECT RESPONSES
    # (We only analyze reaction times for correct answers, since wrong answers
    #  might reflect different cognitive processes like confusion or guessing)
    correct_df = df[df['correct'] == True].copy()
    
    if correct_df.empty:
        print("No correct responses to analyze")
        return
    
    # BASIC STATISTICS
    print("=" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total trials: {len(df)}")
    print(f"Correct responses: {len(correct_df)} ({len(correct_df)/len(df)*100:.1f}%)")
    print(f"Mean reaction time: {correct_df['reaction_time'].mean():.2f} ms")
    print(f"Std reaction time: {correct_df['reaction_time'].std():.2f} ms")
    
    # THE DISTANCE EFFECT ANALYSIS
    # Group trials by the numerical difference between the digits
    by_difference = correct_df.groupby('difference')['reaction_time'].agg(['mean', 'std', 'count'])
    print("\n" + "=" * 50)
    print("DISTANCE EFFECT ANALYSIS")
    print("=" * 50)
    print("Reaction time by numerical difference:")
    print("(The key finding: smaller differences should have longer reaction times)")
    print(by_difference)
    
    # CORRELATION ANALYSIS (replicating the original 1967 study)
    print("\n" + "=" * 50)
    print("CORRELATIONS WITH REACTION TIME")
    print("=" * 50)
    print("(These replicate the statistical tests from the original paper)")
    print(f"Smaller digit (S): r = {correct_df['reaction_time'].corr(correct_df['smaller_digit']):.3f}")
    print(f"Difference (L-S): r = {correct_df['reaction_time'].corr(correct_df['difference']):.3f}")
    
    # WELFORD'S EQUATION TEST
    # This is the mathematical formula that Moyer & Landauer found fits the data
    correct_df['log_ratio'] = np.log(correct_df['larger_digit'] / correct_df['difference'])
    print(f"Log[L/(L-S)]: r = {correct_df['reaction_time'].corr(correct_df['log_ratio']):.3f}")
    print("\nNote: Log[L/(L-S)] should have the highest correlation if Welford's equation fits")
    
    # CREATE VISUALIZATIONS
    print("\n" + "=" * 50)
    print("CREATING GRAPHS...")
    print("=" * 50)
    
    plt.figure(figsize=(12, 8))  # Create a large figure with multiple plots
    
    # PLOT 1: THE DISTANCE EFFECT
    # This should show that larger differences = faster responses
    plt.subplot(2, 2, 1)  # Top-left plot
    by_diff_means = by_difference['mean']
    plt.plot(by_diff_means.index, by_diff_means.values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Numerical Difference (L - S)')
    plt.ylabel('Mean Reaction Time (ms)')
    plt.title('The Distance Effect\n(Key Finding: Larger differences = Faster responses)')
    plt.grid(True, alpha=0.3)
    
    # PLOT 2: REACTION TIME vs SMALLER DIGIT
    plt.subplot(2, 2, 2)  # Top-right plot
    by_smaller = correct_df.groupby('smaller_digit')['reaction_time'].mean()
    plt.plot(by_smaller.index, by_smaller.values, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Smaller Digit (S)')
    plt.ylabel('Mean Reaction Time (ms)')
    plt.title('RT vs Smaller Digit\n(Tests one theory about what drives reaction time)')
    plt.grid(True, alpha=0.3)
    
    # PLOT 3: WELFORD'S EQUATION FIT
    plt.subplot(2, 2, 3)  # Bottom-left plot
    plt.scatter(correct_df['log_ratio'], correct_df['reaction_time'], alpha=0.6, s=30)
    plt.xlabel('Log[L/(L-S)]')
    plt.ylabel('Reaction Time (ms)')
    plt.title('Welford Equation Fit\n(Tests if physical and symbolic comparisons follow same law)')
    plt.grid(True, alpha=0.3)
    
    # PLOT 4: ACCURACY BY DIFFERENCE
    plt.subplot(2, 2, 4)  # Bottom-right plot
    accuracy_by_diff = df.groupby('difference')['correct'].mean()
    plt.plot(accuracy_by_diff.index, accuracy_by_diff.values, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Numerical Difference (L - S)')
    plt.ylabel('Accuracy (Proportion Correct)')
    plt.title('Accuracy by Difference\n(Should show that larger differences = more accurate)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Set y-axis from 0 to 1 (0% to 100%)
    
    # SAVE AND SHOW THE GRAPHS
    plt.tight_layout()  # Make sure plots don't overlap
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')  # Save high-quality image
    print("Graphs saved as 'experiment_results.png'")
    plt.show()  # Display the graphs on screen

def main():
    """
    MAIN FUNCTION - The starting point of the program
    
    This function runs when you execute the script. It sets everything up,
    runs the experiment, and analyzes the results.
    """
    
    # WELCOME MESSAGE
    print("=" * 60)
    print("DIGIT COMPARISON EXPERIMENT")
    print("Based on Moyer & Landauer (1967)")
    print("=" * 60)
    print("Starting in fullscreen mode...")
    print("Press ESC at any time to quit the experiment")
    print("Results will be saved to 'experiment_results.csv'")
    print("=" * 60)
    
    # CREATE AND RUN THE EXPERIMENT
    experiment = DigitComparisonExperiment(fullscreen=True)
    
    try:
        # RUN THE EXPERIMENT AND GET THE DATA
        results_df = experiment.run_experiment()
        
        if results_df is not None:
            # SAVE THE RAW DATA
            results_df.to_csv('experiment_results.csv', index=False)
            print("\n" + "=" * 60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("Results saved to 'experiment_results.csv'")
            print("You can open this file in Excel or other spreadsheet programs")
            
            # ANALYZE THE RESULTS
            analyze_results(results_df)
            
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            print("Check the graphs and statistics above to see if you replicated")
            print("the famous 'distance effect' discovered by Moyer & Landauer!")
        
    except Exception as e:
        print(f"\nError running experiment: {e}")
        print("The experiment encountered an unexpected problem.")
    
    finally:
        pygame.quit()  # Clean up and close the graphics system
        print("\nExperiment window closed. Thank you!")

# PROGRAM ENTRY POINT
# This is where the program actually starts running when you execute the file
if __name__ == "__main__":
    main()  # Call the main function to start everything