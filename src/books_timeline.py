import re

from src.tag import BookTag

RATING_RE = re.compile(r"(\d+(\.\d+)?)\/10")


TOGGLE_BUTTON_HTML = """<div class="toggle-btn" onclick="toggleNote('note-{idx}')">Show Note</div>
<div class="note" id="note-{idx}">{book_body}</div>"""


def get_timeline_html(book_tags: list[BookTag]) -> str:
    html_content = """
    <style>
        body {
            background-color: #2b2b2b;
            color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .timeline {
            position: relative;
            margin: 20px 0;
            padding: 0 40px;
            border-left: 3px solid #555;
        }
        .entry {
            position: relative;
            margin: 20px 0;
            padding-left: 20px;
        }
        .entry:before {
            content: "";
            position: absolute;
            left: -12px;
            top: 5px;
            width: 10px;
            height: 10px;
            background-color: #2b2b2b;
            border: 3px solid #f39c12;
            border-radius: 50%;
            box-shadow: 0 0 5px #f39c12;
        }
        .date {
            font-weight: bold;
            color: #f39c12;
            margin-bottom: 5px;
        }
        .title {
            font-size: 1.1em;
            color: #7fffd4;
            margin: 5px 0;
        }
        .extra {
            font-size: 0.9em;
            color: #00ffff;
        }
        .note {
            font-style: italic;
            color: #bdc3c7;
            display: none;
            margin-top: 10px;
        }
        .toggle-btn {
            color: #3498db;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .toggle-btn:hover {
            text-decoration: underline;
        }
    </style>
    <div class="timeline">
    """

    for idx, book in enumerate(book_tags):
        formatted_date = book.full_date.strftime("%B %d, %Y")
        rating = book.rating
        num_pages = book.number_of_pages
        rating_info = f"{rating:.1f}/10" if rating else ""
        num_pages_info = f"[{num_pages}pg]" if num_pages else ""
        rating_num_pages_div = (
            f'<div class="extra">{rating_info} {num_pages_info}</div>'
            if rating_info or num_pages_info
            else ""
        )
        new_body = RATING_RE.sub("", book.body).strip()
        button_logic = (
            TOGGLE_BUTTON_HTML.format(idx=idx, book_body=new_body) if new_body else ""
        )
        html_content += f"""
        <div class="entry">
            <div class="date">{formatted_date}</div>
            <div class="title">{book.title}</div>
            {rating_num_pages_div}
            {button_logic}
        </div>
        """

    html_content += """
    </div>
    <script>
        function toggleNote(noteId) {
            const note = document.getElementById(noteId);
            const btn = note.previousElementSibling;
            if (note.style.display === "none" || note.style.display === "") {
                note.style.display = "block";
                btn.textContent = "Hide Note";
            } else {
                note.style.display = "none";
                btn.textContent = "Show Note";
            }
    }
    </script>
    """

    return html_content
