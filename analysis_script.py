"""
DISTANCE EFFECT ANALYSIS SCRIPT
Analyzes data from the Digit Comparison Experiment and creates a publication-quality graph

Usage: python analyze_distance_effect.py [filename]
If no filename provided, defaults to 'experiment_results.csv'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats

def load_data(filename='experiment_results.csv'):
    """Load the experiment data from CSV file"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Successfully loaded {filename}")
        print(f"  Total trials: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: Could not find '{filename}'")
        print("  Make sure the file is in the same directory as this script")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

def analyze_and_plot(df, output_filename='distance_effect_results.png'):
    """
    Analyze the distance effect and create a comprehensive single-panel graph
    """
    
    # Filter to correct responses only
    correct_df = df[df['correct'] == True].copy()
    
    if len(correct_df) == 0:
        print("✗ Error: No correct responses found in data")
        return
    
    # Calculate statistics
    total_trials = len(df)
    correct_trials = len(correct_df)
    accuracy = (correct_trials / total_trials) * 100
    mean_rt = correct_df['reaction_time'].mean()
    
    # Group by numerical difference
    by_difference = correct_df.groupby('difference').agg({
        'reaction_time': ['mean', 'std', 'count']
    })
    by_difference.columns = ['mean_rt', 'std_rt', 'count']
    by_difference['sem'] = by_difference['std_rt'] / np.sqrt(by_difference['count'])
    
    # Calculate correlation (key statistic from Moyer & Landauer)
    correlation = correct_df['difference'].corr(correct_df['reaction_time'])
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        by_difference.index, 
        by_difference['mean_rt']
    )
    
    # Create the figure
    plt.figure(figsize=(10, 7))
    
    # Main plot: Distance Effect
    plt.errorbar(by_difference.index, by_difference['mean_rt'], 
                 yerr=by_difference['sem'],
                 fmt='o-', color='#2E86AB', linewidth=2.5, markersize=10,
                 capsize=5, capthick=2, elinewidth=2,
                 label='Mean RT ± SEM')
    
    # Add regression line
    x_line = np.array([by_difference.index.min(), by_difference.index.max()])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, '--', color='#A23B72', linewidth=2, 
             alpha=0.7, label=f'Linear fit (r = {r_value:.3f})')
    
    # Formatting
    plt.xlabel('Numerical Distance', fontsize=14, fontweight='bold')
    plt.ylabel('Reaction Time (ms)', fontsize=14, fontweight='bold')
    plt.title('The Distance Effect in Numerical Cognition\n' + 
              'Replication of Moyer & Landauer (1967)',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # Add statistics text box
    stats_text = (
        f'N = {correct_trials} trials\n'
        f'Accuracy = {accuracy:.1f}%\n'
        f'Mean RT = {mean_rt:.0f} ms\n'
        f'Correlation: r = {correlation:.3f}\n'
        f'Slope: {slope:.2f} ms per unit\n'
        f'p < {p_value:.4f}' if p_value < 0.05 else f'p = {p_value:.4f}'
    )
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add interpretation note
    interpretation = (
        'Key Finding: Reaction time decreases as numerical distance increases.\n'
        'This supports analog magnitude representation of numbers.'
    )
    
    plt.text(0.5, -0.15, interpretation,
             transform=plt.gca().transAxes,
             fontsize=10,
             ha='center',
             style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved as '{output_filename}'")
    
    # Display the figure
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DISTANCE EFFECT ANALYSIS")
    print("="*60)
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct_trials}/{total_trials} trials)")
    print(f"  Mean RT (correct): {mean_rt:.2f} ms (SD = {correct_df['reaction_time'].std():.2f})")
    
    print(f"\nDistance Effect:")
    print(f"  Correlation (distance × RT): r = {correlation:.3f}, p < {p_value:.4f}")
    print(f"  Linear slope: {slope:.2f} ms per unit distance")
    print(f"  RT reduction per unit: {abs(slope):.2f} ms")
    
    print(f"\nMean RT by Distance:")
    print(by_difference[['mean_rt', 'std_rt', 'count']].to_string())
    
    rt_range = by_difference['mean_rt'].max() - by_difference['mean_rt'].min()
    print(f"\n  RT range: {rt_range:.2f} ms (from distance 1 to {by_difference.index.max()})")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if correlation < -0.5 and p_value < 0.05:
        print("✓ Strong distance effect detected!")
        print("  Results replicate Moyer & Landauer (1967)")
        print("  Supports analog magnitude representation")
    elif correlation < 0 and p_value < 0.05:
        print("✓ Distance effect detected")
        print("  Negative correlation confirms the effect")
    else:
        print("⚠ Distance effect not clearly demonstrated")
        print("  Consider collecting more data or checking procedure")
    
    print("="*60 + "\n")

def main():
    """Main function to run the analysis"""
    
    print("\n" + "="*60)
    print("DISTANCE EFFECT ANALYSIS")
    print("Moyer & Landauer (1967) Replication")
    print("="*60 + "\n")
    
    # Get filename from command line argument or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'experiment_results.csv'
    
    # Load and analyze data
    df = load_data(filename)
    analyze_and_plot(df)

if __name__ == "__main__":
    main()