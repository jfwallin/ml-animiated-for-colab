"""
Answer Collection System for DATA 1010 Labs
Google Colab Compatible

This module provides an easy-to-use system for collecting, saving, and exporting
student answers in Colab notebooks with timestamps.

Usage in notebook:
    from answer_collection_system import AnswerCollector

    collector = AnswerCollector(lab_number=1, group_code=group_code)
    collector.display_question("Q1", "What is the meaning of loss?", rows=4)
"""

import json
from datetime import datetime
from ipywidgets import Textarea, Button, VBox, HBox, HTML, Output, Layout
from IPython.display import display, clear_output
import warnings

class AnswerCollector:
    """
    Manages student answer collection for DATA 1010 labs.

    Features:
    - Interactive answer boxes
    - Automatic timestamping
    - Edit and save functionality
    - Progress tracking
    - Export to JSON and TXT
    """

    def __init__(self, lab_number, group_code, lab_title=""):
        """
        Initialize the answer collector.

        Args:
            lab_number: Lab number (e.g., 1, 2, 3)
            group_code: Unique group identifier
            lab_title: Optional descriptive title
        """
        self.lab_number = lab_number
        self.group_code = group_code
        self.lab_title = lab_title

        self.data = {
            "metadata": {
                "lab_number": lab_number,
                "lab_title": lab_title,
                "group_code": group_code,
                "started_at": datetime.now().isoformat(),
                "completed_at": None
            },
            "answers": {},
            "timestamps": {},
            "group_parameters": {}
        }

        self.question_texts = {}  # Store question text for export

    def add_group_parameters(self, params_dict):
        """
        Store group-specific parameters (e.g., true_m, true_b).

        Args:
            params_dict: Dictionary of parameter names and values
        """
        self.data["group_parameters"].update(params_dict)

    def display_question(self, question_id, question_text, rows=3, width='95%'):
        """
        Display an interactive answer box for a question.

        Args:
            question_id: Unique ID (e.g., "Q1", "Q2")
            question_text: The question to display
            rows: Number of rows for text area
            width: Width of text area
        """
        # Store question text
        self.question_texts[question_id] = question_text

        # Question display with nice styling
        q_html = HTML(
            f"<div style='background:#e8f0fe; padding:12px; margin:15px 0; "
            f"border-left:5px solid #1967d2; border-radius:4px;'>"
            f"<span style='color:#1967d2; font-weight:bold; font-size:14px;'>"
            f"Question {question_id.replace('Q', '')}</span><br/>"
            f"<span style='color:#202124; font-size:14px; line-height:1.5;'>"
            f"{question_text}</span>"
            f"</div>"
        )

        # Get existing answer if any
        existing_answer = self.data["answers"].get(question_id, "")

        # Answer text area
        answer_box = Textarea(
            value=existing_answer,
            placeholder="Type your group's answer here... (Discuss first, then type!)",
            layout=Layout(width=width, height=f'{rows*35}px'),
            style={'description_width': '0px'}
        )

        # Save button
        is_saved = question_id in self.data["timestamps"]
        save_btn = Button(
            description="💾 Save Answer" if not is_saved else "✓ Saved",
            button_style='success' if is_saved else 'info',
            tooltip='Click to save your answer with timestamp',
            layout=Layout(width='150px'),
            icon='check' if is_saved else 'save'
        )

        # Status display
        status_html = HTML("")
        if is_saved:
            saved_time = datetime.fromisoformat(self.data["timestamps"][question_id])
            status_html.value = (
                f"<span style='color:#188038; font-size:12px; margin-left:10px;'>"
                f"✓ Last saved: {saved_time.strftime('%I:%M:%S %p')}</span>"
            )

        def on_save(btn):
            """Save answer with timestamp."""
            answer_text = answer_box.value.strip()

            if not answer_text:
                status_html.value = (
                    "<span style='color:#ea8600; font-size:12px; margin-left:10px;'>"
                    "⚠ Answer is empty - type something first!</span>"
                )
                return

            # Save answer
            self.data["answers"][question_id] = answer_text
            self.data["timestamps"][question_id] = datetime.now().isoformat()

            # Update UI
            save_btn.button_style = 'success'
            save_btn.description = "✓ Saved"
            save_btn.icon = 'check'
            saved_time = datetime.now()
            status_html.value = (
                f"<span style='color:#188038; font-size:12px; margin-left:10px;'>"
                f"✓ Saved at {saved_time.strftime('%I:%M:%S %p')}</span>"
            )

        save_btn.on_click(on_save)

        # Helper text
        helper_html = HTML(
            "<div style='color:#5f6368; font-size:11px; margin-top:5px;'>"
            "💡 Tip: You can edit and re-save your answer anytime before final export"
            "</div>"
        )

        # Combine elements
        display(VBox([
            q_html,
            answer_box,
            HBox([save_btn, status_html]),
            helper_html
        ], layout=Layout(margin='10px 0 20px 0')))

    def show_progress(self):
        """Display progress on answering questions."""
        total_questions = len(self.question_texts)
        if total_questions == 0:
            print("⚠ No questions have been displayed yet")
            return

        answered = len([a for a in self.data["answers"].values() if a.strip()])
        progress_pct = (answered / total_questions) * 100 if total_questions > 0 else 0

        # Progress bar
        progress_html = f"""
        <div style='margin: 20px 0; font-family: sans-serif;'>
            <h3 style='color:#202124; margin-bottom:10px;'>📊 Your Progress</h3>
            <div style='background: #e8eaed; border-radius: 8px; height: 35px;
                        position: relative; overflow:hidden;'>
                <div style='background: linear-gradient(90deg, #1967d2, #188038);
                            width: {progress_pct}%; height: 100%;
                            transition: width 0.3s ease;
                            display: flex; align-items: center; justify-content: center;'>
                    <span style='color: white; font-weight: bold; font-size:14px;'>
                        {answered}/{total_questions} ({progress_pct:.0f}%)
                    </span>
                </div>
            </div>
            <p style='margin-top: 12px; color:#5f6368; font-size:13px;'>
                ✓ Answered: <b>{answered}</b> questions<br/>
                ⏳ Remaining: <b>{total_questions - answered}</b> questions
            </p>
        </div>
        """

        display(HTML(progress_html))

        # Show missing questions
        if answered < total_questions:
            missing = []
            for q_id in self.question_texts.keys():
                if q_id not in self.data["answers"] or not self.data["answers"][q_id].strip():
                    missing.append(q_id)

            if missing:
                missing_str = ', '.join(sorted(missing, key=lambda x: int(x.replace('Q','')))[:10])
                print(f"⚠ Unanswered questions: {missing_str}")
                if len(missing) > 10:
                    print(f"   ... and {len(missing) - 10} more")

    def export_answers(self):
        """
        Generate downloadable answer files.

        Creates both TXT (human-readable) and JSON (machine-readable) files.
        """
        # Check if running in Colab
        try:
            from google.colab import files
            in_colab = True
        except ImportError:
            in_colab = False
            warnings.warn("Not running in Colab - files will be saved locally only")

        # Update completion timestamp
        self.data["metadata"]["completed_at"] = datetime.now().isoformat()

        # Count answered questions
        total = len(self.question_texts)
        answered = len([a for a in self.data["answers"].values() if a.strip()])

        print(f"Generating answer files for Lab {self.lab_number}, Group {self.group_code}...")
        print(f"Questions answered: {answered}/{total}")

        if answered < total:
            print(f"\n⚠ Warning: {total - answered} questions are unanswered")
            print("  You can still export, but consider completing all questions first")

        # Generate human-readable text file
        text_output = self._generate_text_format(answered, total)

        # Generate JSON
        json_output = json.dumps(self.data, indent=2)

        # Filenames
        txt_filename = f"Lab{self.lab_number}_Answers_Group{self.group_code}.txt"
        json_filename = f"Lab{self.lab_number}_Answers_Group{self.group_code}.json"

        # Save files
        with open(txt_filename, "w", encoding='utf-8') as f:
            f.write(text_output)

        with open(json_filename, "w", encoding='utf-8') as f:
            f.write(json_output)

        print(f"\n✅ Files generated successfully!")
        print(f"   1. {txt_filename} (human-readable)")
        print(f"   2. {json_filename} (for grading system)")

        # Create download buttons if in Colab
        if in_colab:
            print("\nClick the buttons below to download your files:")

            download_txt_btn = Button(
                description="📥 Download TXT",
                button_style='primary',
                layout=Layout(width='180px', height='40px'),
                tooltip='Download human-readable version'
            )

            download_json_btn = Button(
                description="📥 Download JSON",
                button_style='success',
                layout=Layout(width='180px', height='40px'),
                tooltip='Download for LMS submission'
            )

            output_area = Output()

            def download_txt(b):
                with output_area:
                    clear_output(wait=True)
                    files.download(txt_filename)
                    print(f"✓ Downloaded {txt_filename}")

            def download_json(b):
                with output_area:
                    clear_output(wait=True)
                    files.download(json_filename)
                    print(f"✓ Downloaded {json_filename}")

            download_txt_btn.on_click(download_txt)
            download_json_btn.on_click(download_json)

            display(HBox([download_txt_btn, download_json_btn],
                        layout=Layout(margin='10px 0')))
            display(output_area)
        else:
            print(f"\n📁 Files saved to current directory")

        # Show preview
        self._show_preview(text_output)

        # Submission instructions
        self._show_submission_instructions()

        return txt_filename, json_filename

    def _generate_text_format(self, answered, total):
        """Generate human-readable text format."""
        output = f"""
{'='*80}
DATA 1010 - Lab {self.lab_number}: {self.lab_title}
{'='*80}

GROUP INFORMATION:
  Group Code: {self.group_code}
  Lab Started: {self.data['metadata']['started_at']}
  Lab Completed: {self.data['metadata']['completed_at']}
  Questions Answered: {answered}/{total}

{'='*80}
ANSWERS:
{'='*80}

"""

        # Sort question IDs numerically
        question_ids = sorted(
            self.question_texts.keys(),
            key=lambda x: int(x.replace('Q', ''))
        )

        # Add each answer
        for q_id in question_ids:
            timestamp = self.data["timestamps"].get(q_id, "Not answered")
            answer = self.data["answers"].get(q_id, "[No answer provided]")
            question_text = self.question_texts[q_id]

            output += f"\n{q_id}: {question_text}\n"
            output += f"Timestamp: {timestamp}\n"
            output += f"Answer:\n{answer}\n"
            output += f"{'-'*80}\n"

        # Add group parameters if any
        if self.data["group_parameters"]:
            output += f"\n\n{'='*80}\n"
            output += "GROUP-SPECIFIC PARAMETERS:\n"
            output += "(For instructor verification - these were revealed at end of lab)\n"
            output += f"{'='*80}\n"
            for key, value in self.data["group_parameters"].items():
                if isinstance(value, (int, float)):
                    output += f"  {key}: {value:.6f}\n"
                else:
                    output += f"  {key}: {value}\n"

        output += f"\n{'='*80}\n"
        output += f"End of Lab {self.lab_number} Answers\n"
        output += f"{'='*80}\n"

        return output

    def _show_preview(self, text_output):
        """Show preview of answers."""
        preview_html = f"""
        <div style='margin:20px 0; padding:15px; background:#f8f9fa;
                    border:1px solid #dadce0; border-radius:8px;'>
            <h4 style='color:#202124; margin-top:0;'>📄 Preview of Your Answers:</h4>
            <pre style='font-size:11px; color:#5f6368; max-height:300px;
                        overflow-y:auto; white-space:pre-wrap;'>
{text_output[:1500]}
            </pre>
            {'<i>... (preview truncated, full content in downloaded file)</i>' if len(text_output) > 1500 else ''}
        </div>
        """
        display(HTML(preview_html))

    def _show_submission_instructions(self):
        """Display submission instructions."""
        instructions_html = """
        <div style='background:#fff3cd; border-left:5px solid #ffc107;
                    padding:15px; margin:20px 0; border-radius:4px;'>
            <h4 style='color:#856404; margin-top:0;'>📤 Submission Instructions</h4>
            <ol style='color:#856404; line-height:1.8;'>
                <li><b>Download BOTH files</b> (TXT and JSON) using the buttons above</li>
                <li><b>Submit the JSON file</b> to your course LMS/Canvas</li>
                <li><b>Keep the TXT file</b> for your own records</li>
                <li><b>Verify the file name</b> includes your correct group code</li>
                <li>If you need to make changes, re-run the export cell after editing</li>
            </ol>
            <p style='color:#856404; margin-bottom:0;'>
                <b>File naming format:</b> Lab{self.lab_number}_Answers_Group{self.group_code}.json
            </p>
        </div>
        """
        display(HTML(instructions_html))


# Convenience function for quick setup
def create_collector(lab_number, group_code, lab_title=""):
    """
    Quick setup function to create an answer collector.

    Example:
        collector = create_collector(1, 1234, "Models and Optimization")
        collector.display_question("Q1", "What is loss?")
    """
    return AnswerCollector(lab_number, group_code, lab_title)


# Example usage for instructors
if __name__ == "__main__":
    print("Answer Collection System for DATA 1010")
    print("="*50)
    print("\nUsage in notebook:")
    print("""
    from answer_collection_system import AnswerCollector

    # Initialize
    collector = AnswerCollector(
        lab_number=1,
        group_code=group_code,
        lab_title="Models, Errors, Loss, Optimization"
    )

    # Display questions
    collector.display_question("Q1", "What is loss?", rows=4)

    # Check progress
    collector.show_progress()

    # Add group parameters before export
    collector.add_group_parameters({
        "true_m": true_m,
        "true_b": true_b
    })

    # Export answers
    collector.export_answers()
    """)
