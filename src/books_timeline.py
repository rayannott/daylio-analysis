from src.tag import BookTag
from src.clippings import Highlight


TOGGLE_BUTTON_HTML = """<div class="toggle-btn" onclick="toggleNote('note-{idx}')">Show Note</div>
<div class="note" id="note-{idx}">{book_body}</div>"""


def get_highlights_html(highlights: list[Highlight], idx: int) -> str:
    """Generates HTML for the collapsable list of highlights for a book without title/author and added_on info."""
    if not highlights:
        return ""

    highlights_html = '<div class="highlights">'
    for h in highlights:
        # Only include page and location details
        page_info = f" | page: {h.page}" if h.page is not None else ""
        location_info = f"location: {h.location[0]}-{h.location[1]}"
        date_time = f"on {h.added_on:%d.%m.%Y} at {h.added_on:%H:%M}"

        highlight_html = f"""
        <div class="highlight">
            <div class="highlight-meta">{location_info} | {date_time}{page_info} </div>
            <div class="highlight-text">{h.text}</div>
        """
        if h.note:
            highlight_html += f'<div class="highlight-note">{h.note}</div>'
        highlight_html += "</div>"
        highlights_html += highlight_html
    highlights_html += "</div>"

    toggle_html = f"""
    <div class="toggle-btn" onclick="toggleHighlights('highlights-{idx}')">Show Highlights</div>
    <div class="highlights-container" id="highlights-{idx}" style="display: none;">
        {highlights_html}
    </div>
    """
    return toggle_html


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
        .note, .highlights-container {
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
        .highlight {
            margin-bottom: 15px;
            padding: 5px;
            border-bottom: 1px solid #555;
        }
        .highlight-meta {
            font-size: 0.85em;
            color: #8ab4f8;
        }
        .highlight-text {
            margin-top: 5px;
        }
        .highlight-note {
            margin-top: 5px;
            font-style: normal;
            color: #d3d3d3;
        }
    </style>
    <div class="timeline">
    """

    for idx, book in enumerate(book_tags):
        formatted_date = book.full_date.strftime("%B %d, %Y")
        rating = book.rating
        num_pages = book.number_of_pages
        rating_info = f"{rating:.1f}/10" if rating is not None else ""
        num_pages_info = f"[{num_pages} pages]" if num_pages is not None else ""
        rating_num_pages_div = (
            f'<div class="extra">{rating_info} {num_pages_info}</div>'
            if rating_info or num_pages_info
            else ""
        )
        button_logic = (
            TOGGLE_BUTTON_HTML.format(idx=idx, book_body=book.body_clean)
        )
        highlights_section = get_highlights_html(book.highlights, idx)

        author_str = f" [{book.author}]" if book.author else ""

        html_content += f"""
        <div class="entry">
            <div class="date">{formatted_date}</div>
            <div class="title">{book.title}{author_str}</div>
            {rating_num_pages_div}
            {button_logic}
            {highlights_section}
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
        function toggleHighlights(highlightsId) {
            const container = document.getElementById(highlightsId);
            const btn = container.previousElementSibling;
            if (container.style.display === "none" || container.style.display === "") {
                container.style.display = "block";
                btn.textContent = "Hide Highlights";
            } else {
                container.style.display = "none";
                btn.textContent = "Show Highlights";
            }
        }
    </script>
    """

    return html_content
