import nbformat
import json
import re
import html
from markdown import markdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain imports
from langchain.tools import BaseTool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

MODEL = os.getenv("MODEL", "minimax-m2.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# ---------------------------
# TOOL 1: Parse Notebook
# ---------------------------
class NotebookParserTool(BaseTool):
    """Parse a notebook file and return normalized cell data as a string."""

    name: str = "notebook_parser"
    description: str = "Parses a .ipynb file and extracts cells"

    def _run(self, file_path: str):
        nb = nbformat.read(file_path, as_version=4)

        parsed_data = []

        for cell in nb.cells:
            cell_data = {
                "type": cell.cell_type,
                "source": cell.source,
                "outputs": []
            }

            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if "text" in output:
                        cell_data["outputs"].append(output["text"])
                    elif "data" in output:
                        if "text/plain" in output["data"]:
                            cell_data["outputs"].append(output["data"]["text/plain"])

            parsed_data.append(cell_data)

        return json.dumps(parsed_data)


# ---------------------------
# TOOL 2: Formatter
# ---------------------------
class FormatterTool(BaseTool):
    """Format parsed notebook content into render-ready blocks."""

    name: str = "formatter"
    description: str = "Formats parsed notebook content"

    def _run(self, parsed_json: str | list):
        """Convert parsed notebook text into markdown/code/output tuples.

        Args:
            parsed_json: Stringified parsed notebook structure.

        Returns:
            String representation of formatted content tuples.
        """
        # LangGraph may pass tool output as an already-parsed list.
        if isinstance(parsed_json, (list, dict)):
            parsed_data = parsed_json
        else:
            parsed_data = json.loads(parsed_json)

        formatted = []

        for cell in parsed_data:
            if cell["type"] == "markdown":
                formatted.append({
                    "type": "markdown",
                    # Keep raw markdown so heading/list structure can be rendered in PDF.
                    "content": cell["source"],
                })

            elif cell["type"] == "code":
                formatted.append({
                    "type": "code",
                    "content": cell["source"]
                })

                for out in cell["outputs"]:
                    formatted.append({
                        "type": "output",
                        "content": out
                    })

        return json.dumps(formatted)


# ---------------------------
# TOOL 3: PDF Generator
# ---------------------------
class PDFGeneratorTool(BaseTool):
    """Generate a PDF report from formatted notebook content."""

    name: str = "pdf_generator"
    description: str = "Generates a PDF from formatted content"

    @staticmethod
    def _split_markdown_blocks(md_text: str) -> list[str]:
        blocks = []
        current = []
        for line in md_text.splitlines():
            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                continue
            current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
        return blocks

    @staticmethod
    def _render_markdown_block(block: str, styles) -> list:
        flowables = []
        lines = block.splitlines()
        if not lines:
            return flowables

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", lines[0])
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = html.escape(heading_match.group(2).strip())
            style_name = {1: "Heading1", 2: "Heading2", 3: "Heading3"}[level]
            flowables.append(Paragraph(heading_text, styles[style_name]))

            # Render any remaining lines below the heading as normal markdown text.
            remaining = "\n".join(lines[1:]).strip()
            if remaining:
                html_text = markdown(remaining)
                html_text = re.sub(r"^<p>|</p>$", "", html_text.strip())
                flowables.append(Paragraph(html_text, styles["Normal"]))
                flowables.append(Spacer(1, 4))

            return flowables

        is_list = True
        for line in lines:
            if not re.match(r"^\s*([-*+]\s+|\d+\.\s+)", line):
                is_list = False
                break

        if is_list:
            for line in lines:
                item_text = re.sub(r"^\s*([-*+]\s+|\d+\.\s+)", "", line).strip()
                html_item = markdown(item_text)
                html_item = re.sub(r"^<p>|</p>$", "", html_item.strip())
                flowables.append(Paragraph(html_item, styles["List"], bulletText="•"))
            flowables.append(Spacer(1, 4))
            return flowables

        html_text = markdown(block)
        html_text = re.sub(r"^<p>|</p>$", "", html_text.strip())
        flowables.append(Paragraph(html_text, styles["Normal"]))
        flowables.append(Spacer(1, 4))
        return flowables

    def _run(self, formatted_json: str | list):
        """Build output.pdf from formatted markdown/code/output tuples.

        Args:
            formatted_json: Stringified list of content tuples.

        Returns:
            A status message with the generated PDF filename.
        """
        # Accept either raw JSON text or already-parsed content.
        if isinstance(formatted_json, (list, dict)):
            content = formatted_json
        else:
            content = json.loads(formatted_json)

        doc = SimpleDocTemplate("output.pdf")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("List", parent=styles["Normal"], leftIndent=14, bulletIndent=4, leading=14))
        styles.add(
            ParagraphStyle(
                "CodeBlock",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=9,
                leading=11,
                backColor=colors.HexColor("#F5F7FA"),
                borderColor=colors.HexColor("#D0D7DE"),
                borderWidth=0.6,
                borderPadding=6,
                spaceBefore=4,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                "OutputBlock",
                parent=styles["Code"],
                fontName="Courier",
                fontSize=8.5,
                leading=10.5,
                backColor=colors.HexColor("#F9FBFC"),
                borderColor=colors.HexColor("#C9D1D9"),
                borderWidth=0.6,
                borderPadding=6,
                spaceBefore=2,
                spaceAfter=8,
            )
        )

        elements = []

        elements.append(Paragraph("Notebook Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Generated from Jupyter notebook content.", styles["Normal"]))
        elements.append(Spacer(1, 24))

        for item in content:
            if item["type"] == "markdown":
                blocks = self._split_markdown_blocks(item["content"])
                for block in blocks:
                    elements.extend(self._render_markdown_block(block, styles))

            elif item["type"] == "code":
                elements.append(Paragraph("Code:", styles["Heading4"]))
                elements.append(Preformatted(item["content"], styles["CodeBlock"]))

            elif item["type"] == "output":
                elements.append(Paragraph("Output:", styles["Heading5"]))
                elements.append(Preformatted(str(item["content"]), styles["OutputBlock"]))

            elements.append(Spacer(1, 12))

        doc.build(elements)

        return "PDF generated as output.pdf"


# ---------------------------
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
)


# ---------------------------
# CREATE AGENT
# ---------------------------
tools = [
    NotebookParserTool(),
    FormatterTool(),
    PDFGeneratorTool()
]

agent = create_agent(
    llm,
    tools=tools,
    system_prompt="You are a helpful assistant for converting Jupyter notebooks into PDF reports. Follow the instructions step by step. If user provides other files (not Jupyter notebooks .ipynb files) to convert simply respond with 'Unsupported file type.' "
)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    file_path = "notebook.ipynb"

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": f"Convert this notebook {file_path} into a PDF report and also explain the notebook's content."}
        ]
    })
    print(result["messages"][-1].content)