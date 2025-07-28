# main.py
import os
import argparse
import json
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file (for API key)
load_dotenv()

# Configure the Gemini API key
# IMPORTANT: Create a .env file in the same directory and add your key like this:
# GOOGLE_API_KEY="AIzaSyCiWynS8RZ-wAgCxHU4ZoOjv8_9PV9txso"
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please make sure you have a .env file with your GOOGLE_API_KEY.")
    exit()


# --- Core Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error reading PDF {os.path.basename(pdf_path)}: {e}")
        return None

def get_details_from_gemini(resume_text: str) -> dict:
    """
    Uses the Gemini API to extract structured data from resume text.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # The prompt is carefully designed to ask for a JSON output.
    prompt = """
    Based on the following resume text, please extract the following information in a valid JSON format.
    If a field is not found, please use "N/A" for strings, [] for lists, or 0 for numbers.

    1.  **name**: The full name of the candidate.
    2.  **college_name**: The name of their primary college or university.
    3.  **positions_of_responsibility**: A list of significant positions of responsibility held.
    4.  **tech_stack**: A list of technologies, programming languages, and tools mentioned.
    5.  **experience_in_years**: The total years of professional experience as a number. Infer this from the text.
    6.  **github_contributions_last_year**: The number of GitHub contributions in the last year, if mentioned. If not mentioned, return 0.

    Resume Text:
    ---
    {resume_text}
    ---

    Please provide the output as a single JSON object.
    """

    try:
        response = model.generate_content(prompt.format(resume_text=resume_text))
        
        # Clean the response to extract only the JSON part
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON string into a Python dictionary
        return json.loads(json_str)
    except Exception as e:
        print(f"An error occurred with the Gemini API call: {e}")
        print(f"Problematic response text: {response.text if 'response' in locals() else 'No response'}")
        return None


def process_resumes(folder_path: str, output_csv: str):
    """
    Processes all PDFs in a folder, extracts data, and saves to a CSV.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The specified folder does not exist: {folder_path}")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return

    console = Console()
    all_resume_data = []

    with console.status("[bold green]Processing resumes...") as status:
        for i, filename in enumerate(pdf_files):
            status.update(f"[bold green]Processing file {i+1}/{len(pdf_files)}: {filename}")
            pdf_path = os.path.join(folder_path, filename)

            # 1. Extract text
            resume_text = extract_text_from_pdf(pdf_path)
            if not resume_text:
                continue

            # 2. Get details using Gemini
            extracted_data = get_details_from_gemini(resume_text)
            if not extracted_data:
                console.log(f"[yellow]Could not process {filename} with LLM.[/yellow]")
                continue

            # 3. Add filename and store
            extracted_data['filename'] = filename
            all_resume_data.append(extracted_data)
            console.log(f"[green]Successfully processed {filename}[/green]")

    # 4. Save to DataFrame and CSV
    if all_resume_data:
        df = pd.DataFrame(all_resume_data)
        
        # Reorder columns for better readability
        cols_order = ['filename', 'name', 'experience_in_years', 'college_name', 'tech_stack', 'positions_of_responsibility', 'github_contributions_last_year']
        df = df[cols_order]

        df.to_csv(output_csv, index=False)
        console.print(f"\n[bold blue]All resumes processed and saved to {output_csv}[/bold blue]")
    else:
        console.print("\n[bold red]No data was extracted from any of the resumes.[/bold red]")


# --- CLI Display and Interaction Functions ---

def display_results(df: pd.DataFrame):
    """Displays the DataFrame in a nice table using rich."""
    if df.empty:
        print("No matching resumes found.")
        return

    table = Table(title="Resume Shortlist")
    for column in df.columns:
        table.add_column(column, style="cyan", no_wrap=False)

    for _, row in df.iterrows():
        # Convert list-like columns to strings for display
        tech_stack = ', '.join(row['tech_stack']) if isinstance(row['tech_stack'], list) else str(row['tech_stack'])
        pors = ', '.join(row['positions_of_responsibility']) if isinstance(row['positions_of_responsibility'], list) else str(row['positions_of_responsibility'])
        
        table.add_row(
            row['filename'],
            row['name'],
            str(row['experience_in_years']),
            row['college_name'],
            tech_stack,
            pors,
            str(row['github_contributions_last_year'])
        )

    console = Console()
    console.print(table)

def search_resumes(csv_file: str, tech: str, exp: float):
    """Searches resumes based on tech stack and experience."""
    try:
        df = pd.read_csv(csv_file)
        # Convert string representation of list back to actual list for searching
        df['tech_stack'] = df['tech_stack'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)
    except FileNotFoundError:
        print(f"Error: The data file '{csv_file}' was not found. Please run the 'process' command first.")
        return

    results = df.copy()
    if tech:
        # Case-insensitive search for technology
        results = results[results['tech_stack'].apply(lambda stack: isinstance(stack, list) and any(tech.lower() in t.lower() for t in stack))]
    
    if exp is not None:
        results = results[results['experience_in_years'] >= exp]

    display_results(results)

def sort_resumes(csv_file: str, sort_by: str, order: str):
    """Sorts resumes by a given column."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The data file '{csv_file}' was not found. Please run the 'process' command first.")
        return

    if sort_by not in df.columns:
        print(f"Error: Invalid column to sort by. Choose from: {', '.join(df.columns)}")
        return

    ascending = order.lower() == 'asc'
    sorted_df = df.sort_values(by=sort_by, ascending=ascending)
    display_results(sorted_df)


# --- Main Execution ---

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="A CLI tool to shortlist resumes using Gemini AI.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Command: process
    process_parser = subparsers.add_parser('process', help='Process PDF resumes from a folder.')
    process_parser.add_argument('folder', type=str, help='Path to the folder containing PDF resumes.')
    process_parser.add_argument('--output', type=str, default='resume_data.csv', help='Name of the output CSV file.')

    # Command: view
    view_parser = subparsers.add_parser('view', help='View all processed resumes.')
    view_parser.add_argument('--file', type=str, default='resume_data.csv', help='Path to the data file.')

    # Command: search
    search_parser = subparsers.add_parser('search', help='Search for resumes.')
    search_parser.add_argument('--file', type=str, default='resume_data.csv', help='Path to the data file.')
    search_parser.add_argument('--tech', type=str, help='Filter by technology (e.g., "Python").')
    search_parser.add_argument('--exp', type=float, help='Filter by minimum years of experience (e.g., 2.5).')

    # Command: sort
    sort_parser = subparsers.add_parser('sort', help='Sort resumes.')
    sort_parser.add_argument('--file', type=str, default='resume_data.csv', help='Path to the data file.')
    sort_parser.add_argument('--by', type=str, required=True, help='Column to sort by (e.g., "experience_in_years").')
    sort_parser.add_argument('--order', type=str, default='desc', choices=['asc', 'desc'], help='Sort order (asc or desc).')

    args = parser.parse_args()

    if args.command == 'process':
        process_resumes(args.folder, args.output)
    elif args.command == 'view':
        try:
            df = pd.read_csv(args.file)
            display_results(df)
        except FileNotFoundError:
            print(f"Error: Data file '{args.file}' not found. Run the 'process' command first.")
    elif args.command == 'search':
        search_resumes(args.file, args.tech, args.exp)
    elif args.command == 'sort':
        sort_resumes(args.file, args.by, args.order)


if __name__ == '__main__':
    main()

