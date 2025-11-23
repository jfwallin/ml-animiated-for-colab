"""
Instructor Grading Helper for DATA 1010 Labs
Processes student answer JSON files for grading

Usage:
    python instructor_grading_helper.py --lab 1 --directory ./submissions

Or in Jupyter/Colab:
    from instructor_grading_helper import GradingHelper
    helper = GradingHelper()
    helper.load_submissions('./submissions')
    helper.create_grading_spreadsheet('lab1_grading.csv')
"""

import json
import glob
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import sys

class GradingHelper:
    """Helper class for processing student lab submissions."""

    def __init__(self, lab_number=None):
        """
        Initialize grading helper.

        Args:
            lab_number: Optional lab number to filter for
        """
        self.lab_number = lab_number
        self.submissions = []

    def load_submissions(self, directory=".", pattern=None):
        """
        Load all JSON answer files from a directory.

        Args:
            directory: Path to directory containing submissions
            pattern: Custom glob pattern (default: Lab*_Answers_Group*.json)

        Returns:
            Number of submissions loaded
        """
        if pattern is None:
            if self.lab_number:
                pattern = f"Lab{self.lab_number}_Answers_Group*.json"
            else:
                pattern = "Lab*_Answers_Group*.json"

        json_files = glob.glob(f"{directory}/{pattern}")

        if not json_files:
            print(f"⚠ No submission files found matching: {directory}/{pattern}")
            return 0

        self.submissions = []
        errors = []

        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.submissions.append({
                        'filepath': filepath,
                        'filename': Path(filepath).name,
                        'data': data
                    })
            except json.JSONDecodeError as e:
                errors.append((filepath, str(e)))
            except Exception as e:
                errors.append((filepath, str(e)))

        print(f"✓ Loaded {len(self.submissions)} submissions")

        if errors:
            print(f"⚠ Failed to load {len(errors)} files:")
            for filepath, error in errors:
                print(f"  - {Path(filepath).name}: {error}")

        return len(self.submissions)

    def create_grading_spreadsheet(self, output_file="grading.csv"):
        """
        Create CSV spreadsheet for grading.

        Args:
            output_file: Path to output CSV file

        Returns:
            pandas DataFrame
        """
        if not self.submissions:
            print("⚠ No submissions loaded. Call load_submissions() first.")
            return None

        print(f"Creating grading spreadsheet...")

        rows = []

        for sub in self.submissions:
            data = sub['data']
            metadata = data.get('metadata', {})
            answers = data.get('answers', {})
            timestamps = data.get('timestamps', {})

            # Basic info
            row = {
                "Filename": sub['filename'],
                "Group Code": metadata.get('group_code', 'Unknown'),
                "Lab Number": metadata.get('lab_number', ''),
                "Started": metadata.get('started_at', ''),
                "Completed": metadata.get('completed_at', ''),
                "Total Answered": len([a for a in answers.values() if a and a.strip()])
            }

            # Calculate duration
            if metadata.get('started_at') and metadata.get('completed_at'):
                try:
                    start = datetime.fromisoformat(metadata['started_at'])
                    end = datetime.fromisoformat(metadata['completed_at'])
                    duration = end - start
                    row["Duration (minutes)"] = round(duration.total_seconds() / 60, 1)
                except:
                    row["Duration (minutes)"] = ""

            # Add each answer as a column
            # Get all question IDs across all submissions
            all_q_ids = set()
            for s in self.submissions:
                all_q_ids.update(s['data'].get('answers', {}).keys())

            # Sort question IDs
            sorted_q_ids = sorted(all_q_ids, key=lambda x: int(x.replace('Q', '')))

            for q_id in sorted_q_ids:
                answer = answers.get(q_id, "")
                row[f"{q_id}_Answer"] = answer

                # Add timestamp if available
                timestamp = timestamps.get(q_id, "")
                if timestamp:
                    try:
                        ts = datetime.fromisoformat(timestamp)
                        row[f"{q_id}_Time"] = ts.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        row[f"{q_id}_Time"] = timestamp

                # Add character count (for quick assessment)
                row[f"{q_id}_Length"] = len(answer.strip()) if answer else 0

            # Add group parameters if available
            params = data.get('group_parameters', {})
            for key, value in params.items():
                row[f"Param_{key}"] = value

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by group code
        if "Group Code" in df.columns:
            df = df.sort_values("Group Code")

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✓ Grading spreadsheet saved: {output_file}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Rows: {len(df)}")

        return df

    def check_completion_stats(self):
        """Display statistics about submission completion."""
        if not self.submissions:
            print("⚠ No submissions loaded")
            return

        print("\n" + "="*60)
        print("COMPLETION STATISTICS")
        print("="*60)

        total = len(self.submissions)

        # Count questions answered
        completion_counts = []
        for sub in self.submissions:
            answers = sub['data'].get('answers', {})
            answered = len([a for a in answers.values() if a and a.strip()])
            total_q = len(answers)
            completion_counts.append((answered, total_q))

        if completion_counts:
            avg_answered = sum(c[0] for c in completion_counts) / len(completion_counts)
            avg_total = sum(c[1] for c in completion_counts) / len(completion_counts)
            print(f"\nAverage completion: {avg_answered:.1f} / {avg_total:.1f} questions")

            # Show distribution
            full_complete = sum(1 for c in completion_counts if c[0] == c[1])
            partial = sum(1 for c in completion_counts if 0 < c[0] < c[1])
            empty = sum(1 for c in completion_counts if c[0] == 0)

            print(f"\nCompletion breakdown:")
            print(f"  Fully complete: {full_complete} ({full_complete/total*100:.1f}%)")
            print(f"  Partially complete: {partial} ({partial/total*100:.1f}%)")
            print(f"  No answers: {empty} ({empty/total*100:.1f}%)")

    def check_submission_times(self):
        """Analyze submission timing and durations."""
        if not self.submissions:
            print("⚠ No submissions loaded")
            return

        print("\n" + "="*60)
        print("SUBMISSION TIMING ANALYSIS")
        print("="*60)

        durations = []

        for sub in self.submissions:
            data = sub['data']
            metadata = data.get('metadata', {})
            group = metadata.get('group_code', 'Unknown')

            if not metadata.get('started_at') or not metadata.get('completed_at'):
                print(f"Group {group}: Missing timestamps")
                continue

            try:
                started = datetime.fromisoformat(metadata['started_at'])
                completed = datetime.fromisoformat(metadata['completed_at'])
                duration = completed - started

                duration_min = duration.total_seconds() / 60

                print(f"Group {group}:")
                print(f"  Started: {started.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Completed: {completed.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Duration: {duration_min:.1f} minutes")

                durations.append(duration_min)

            except Exception as e:
                print(f"Group {group}: Error parsing timestamps - {e}")

        if durations:
            print(f"\nDuration statistics:")
            print(f"  Average: {sum(durations)/len(durations):.1f} minutes")
            print(f"  Minimum: {min(durations):.1f} minutes")
            print(f"  Maximum: {max(durations):.1f} minutes")
            print(f"  Median: {sorted(durations)[len(durations)//2]:.1f} minutes")

    def export_individual_reports(self, output_dir="./grading_reports"):
        """
        Export individual grading reports for each submission.

        Args:
            output_dir: Directory to save individual reports
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Exporting individual reports to {output_dir}...")

        for sub in self.submissions:
            data = sub['data']
            group = data.get('metadata', {}).get('group_code', 'Unknown')

            # Create report
            report = self._generate_individual_report(data)

            # Save report
            filename = f"Report_Group{group}.txt"
            filepath = Path(output_dir) / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)

        print(f"✓ Exported {len(self.submissions)} reports")

    def _generate_individual_report(self, data):
        """Generate a grading report for one submission."""
        metadata = data.get('metadata', {})
        answers = data.get('answers', {})
        timestamps = data.get('timestamps', {})

        report = f"""
{'='*80}
GRADING REPORT - Lab {metadata.get('lab_number', '?')}
{'='*80}

Group Code: {metadata.get('group_code', 'Unknown')}
Submitted: {metadata.get('completed_at', 'Not completed')}

Questions Answered: {len([a for a in answers.values() if a and a.strip()])} / {len(answers)}

{'='*80}
ANSWERS:
{'='*80}

"""

        # Sort questions
        sorted_q_ids = sorted(answers.keys(), key=lambda x: int(x.replace('Q', '')))

        for q_id in sorted_q_ids:
            answer = answers[q_id]
            timestamp = timestamps.get(q_id, 'Not recorded')

            report += f"\n{q_id}:\n"
            report += f"Timestamp: {timestamp}\n"
            report += f"Length: {len(answer)} characters\n"
            report += f"Answer:\n{answer}\n"
            report += f"\nGrade: ____ / ____ \n"
            report += f"Comments:\n\n\n"
            report += f"{'-'*80}\n"

        return report

    def find_similar_answers(self, question_id, threshold=0.8):
        """
        Find potentially plagiarized answers (basic string similarity).

        Args:
            question_id: Question ID to check (e.g., "Q1")
            threshold: Similarity threshold (0-1)

        Note: This is a very basic check. Manual review is always needed.
        """
        from difflib import SequenceMatcher

        if not self.submissions:
            print("⚠ No submissions loaded")
            return

        answers = []
        for sub in self.submissions:
            data = sub['data']
            group = data.get('metadata', {}).get('group_code', 'Unknown')
            answer = data.get('answers', {}).get(question_id, '')
            if answer and answer.strip():
                answers.append((group, answer.strip().lower()))

        print(f"\nChecking {question_id} for similar answers...")
        print(f"(Threshold: {threshold*100:.0f}% similarity)\n")

        similar_pairs = []

        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                group1, ans1 = answers[i]
                group2, ans2 = answers[j]

                similarity = SequenceMatcher(None, ans1, ans2).ratio()

                if similarity >= threshold:
                    similar_pairs.append((group1, group2, similarity))
                    print(f"⚠ Groups {group1} and {group2}: {similarity*100:.1f}% similar")

        if not similar_pairs:
            print(f"✓ No highly similar answers found for {question_id}")

        return similar_pairs


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process DATA 1010 lab submissions for grading"
    )
    parser.add_argument('--lab', type=int, help='Lab number to process')
    parser.add_argument('--directory', default='.', help='Directory containing submissions')
    parser.add_argument('--output', default='grading.csv', help='Output CSV filename')
    parser.add_argument('--stats', action='store_true', help='Show completion statistics')
    parser.add_argument('--timing', action='store_true', help='Show timing analysis')
    parser.add_argument('--reports', action='store_true', help='Generate individual reports')

    args = parser.parse_args()

    # Create helper
    helper = GradingHelper(lab_number=args.lab)

    # Load submissions
    count = helper.load_submissions(args.directory)

    if count == 0:
        print("\n❌ No submissions found. Check the directory and lab number.")
        return

    # Create grading spreadsheet
    print()
    helper.create_grading_spreadsheet(args.output)

    # Show statistics if requested
    if args.stats:
        helper.check_completion_stats()

    if args.timing:
        helper.check_submission_times()

    if args.reports:
        helper.export_individual_reports()

    print("\n✅ Grading processing complete!")


if __name__ == "__main__":
    main()
