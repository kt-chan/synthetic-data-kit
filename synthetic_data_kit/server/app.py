"""
Flask application for the Synthetic Data Kit web interface.
"""

import os, time, json, logging, queue, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    abort,
    flash,
)
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, IntegerField, SelectField, FileField, SubmitField
from wtforms.validators import DataRequired, Optional as OptionalValidator
from urllib.parse import unquote

from synthetic_data_kit.utils.config import (
    load_config,
    get_llm_provider,
    get_config_path,
    get_path_config,
)
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from synthetic_data_kit.core.save_as import convert_format
from synthetic_data_kit.core.ingest import process_file as ingest_process_file
from synthetic_data_kit.utils.app_logger import (
    get_logger,
    add_sse_queue,
    remove_sse_queue,
)

logger = get_logger(__name__)

GLBOAL_TASK_COMPLETED_MESSAGE = "##### Task completed #####"
GLOBAL_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_DEBUG_FLAG = False
GLOBAL_CONFIG = load_config()

app = Flask(
    __name__, static_folder=os.path.join(GLOBAL_BASE_DIR, "static"), static_url_path="/static"
)
app.config["SECRET_KEY"] = os.urandom(24)
executor = ThreadPoolExecutor(max_workers=1)


# Set default paths
DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"
DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "output"
DEFAULT_GENERATED_DIR = DEFAULT_DATA_DIR / "generated"
DEFAULT_CURATED_DIR = DEFAULT_DATA_DIR / "cleaned"
DEFAULT_FINAL_DIR = DEFAULT_DATA_DIR / "final"

# Create directories if they don't exist
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CURATED_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Global task status
task_running = threading.Event()
task_complete = threading.Event()


def reload_config():
    """Reload the configuration from the config file."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = load_config()
    return GLOBAL_CONFIG


def truncate_datastore():
    config = get_config_path()


# Forms
class CreateForm(FlaskForm):
    """Form for creating content from text"""

    input_file = StringField("Input File Path", validators=[DataRequired()])
    content_type = SelectField(
        "Content Type",
        choices=[
            ("qa", "Question-Answer Pairs"),
            ("summary", "Summary"),
            ("cot", "Chain of Thought"),
            ("cot-enhance", "CoT Enhancement"),
        ],
        default="qa",
    )
    num_pairs = IntegerField("Number of QA Pairs", default=100)
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Generate Content")


class ProcessForm(FlaskForm):
    """Form for updating files"""

    editor = TextAreaField("Editor", validators=[DataRequired()])

    def __init__(self, *args, **kwargs):
        super(ProcessForm, self).__init__(*args, **kwargs)
        try:
            # Try to open config.yaml
            with open(DEFAULT_CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
                self.editor.data = f.read()
        except FileNotFoundError as e:
            # If config.yaml does not exist, use config-default.yaml
            with open(DEFAULT_CONFIG_DIR / "config-default.yaml", "r", encoding="utf-8") as f:
                self.editor.data = f.read()

        with open(DEFAULT_CONFIG_DIR / "config-default.yaml", "r", encoding="utf-8") as f:
            self.editor.default = f.read()

    def get_all_files(self, config):

        # Get the list of available input files
        input_files = []
        if DEFAULT_OUTPUT_DIR.exists():
            input_files = [
                str(f.relative_to(DEFAULT_DATA_DIR.parent))
                for f in DEFAULT_OUTPUT_DIR.glob("*.txt")
            ]

        return input_files


class ConfigForm(FlaskForm):
    """Form for updating files"""

    editor = TextAreaField("Editor", validators=[DataRequired()])

    def __init__(self, *args, **kwargs):
        super(ConfigForm, self).__init__(*args, **kwargs)
        try:
            # Try to open config.yaml
            with open(DEFAULT_CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
                self.editor.data = f.read()
        except FileNotFoundError as e:
            # If config.yaml does not exist, use config-default.yaml
            with open(DEFAULT_CONFIG_DIR / "config-default.yaml", "r", encoding="utf-8") as f:
                self.editor.data = f.read()

        with open(DEFAULT_CONFIG_DIR / "config-default.yaml", "r", encoding="utf-8") as f:
            self.editor.default = f.read()


class IngestForm(FlaskForm):
    """Form for ingesting documents"""

    input_type = SelectField(
        "Input Type",
        choices=[("file", "Upload File"), ("url", "URL"), ("path", "Local Path")],
        default="file",
    )
    upload_file = FileField("Upload Document")
    input_path = StringField("File Path or URL")
    output_name = StringField("Output Filename (optional)")
    submit = SubmitField("Parse Document")


class CurateForm(FlaskForm):
    """Form for curating QA pairs"""

    input_file = StringField("Input JSON File Path", validators=[DataRequired()])
    num_pairs = IntegerField("Number of QA Pairs to Keep", default=10)
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Curate QA Pairs")


class UploadForm(FlaskForm):
    """Form for uploading files"""

    file = FileField("Upload File", validators=[DataRequired()])
    submit = SubmitField("Upload")


class SaveAsForm(FlaskForm):
    """Form for generate sft data"""

    input_file = StringField("Input JSON File Path", validators=[DataRequired()])
    output_format = SelectField(
        "Input Type",
        choices=[("ft", "ft"), ("jsonl", "jsonl"), ("alpaca", "alpaca"), ("chatml", "chatml")],
        default="ft",
    )
    storage_format = SelectField(
        "Input Type", choices=[("json", "json"), ("hf", "hf")], default="json"
    )
    submit = SubmitField("Generate SFT Dataset")


@app.template_filter("escape_js")
def escape_js(value):
    # Escape backticks
    value = value.replace("`", "\\`")
    # Escape newlines
    value = value.replace("\\n", " ")
    # Escape double quotes
    value = value.replace('"', '\\"')
    # Escape single quotes
    value = value.replace("'", "\\'")
    return value


# Routes
@app.route("/")
def index():
    """Main index page"""
    provider = get_llm_provider(GLOBAL_CONFIG)
    return render_template("index.html", provider=provider)


@app.route("/process", methods=["GET", "POST"])
def process():
    """Upload a file to the data directory"""

    form = ProcessForm()

    # Call the process_all method when the form is submitted
    files = form.get_all_files(GLOBAL_CONFIG)
    # Optionally, you can redirect or flash a message
    return render_template("process.html", form=form, files=files)


@app.route("/process_all_task", methods=["POST"])
def process_all_task():

    if task_running.is_set():
        return jsonify({"error": "Task is already running"}), 400

    def process_files(files):
        """Background task to process files"""
        global task_running, task_complete
        global GLOBAL_CONFIG
        global GLOBAL_DEBUG_FLAG

        if GLOBAL_CONFIG is None:
            GLOBAL_CONFIG = reload_config()

        provider = get_llm_provider(GLOBAL_CONFIG)
        if provider not in ("vllm", "api-endpoint"):
            flash(f"Error: LLM provider is not defined correctly", "danger")
        time.sleep(2)

        try:
            logger.info(f"Starting to process {len(files)} files")
            for i, file in enumerate(files):
                try:
                    logger.info(f"Processing file {i+1}/{len(files)}: {file}")
                    # Simulate processing time with different log levels
                    time.sleep(1)
                    content_type = "qa"
                    num_pairs = GLOBAL_CONFIG.get("generation", {}).get("num_pairs", 100)
                    filename = Path(file).stem
                    fileformat = Path(file).suffix

                    logger.info(f"Create Task Starting ..., Creating Summary & QA Pairs")
                    create_process = process_file(
                        file_path=file,
                        output_dir=str(DEFAULT_GENERATED_DIR),
                        content_type=content_type,
                        num_pairs=num_pairs,
                        provider=provider,
                        config_path=get_config_path(),
                        verbose=GLOBAL_DEBUG_FLAG,
                    )
                    logger.info(f"Create Task Completed")

                    logger.info(f"Curate TaskStarting ..., Validating QA Pairs")
                    curate_input = DEFAULT_GENERATED_DIR / (filename + "_qa_pairs.json")
                    output_path = DEFAULT_CURATED_DIR / (filename + "_qa_pairs.json")
                    logger.info(f"Executing QA Pairs Curation for file: {curate_input}")
                    curate_process = curate_qa_pairs(
                        input_path=curate_input,
                        output_path=output_path,
                        provider=provider,
                        config_path=get_config_path(),
                        verbose=GLOBAL_DEBUG_FLAG,
                    )
                    logger.info(f"Create Task Completed")

                    logger.info(f"Save Task Starting ...,  Generating SFT Dataset")
                    save_as_input = DEFAULT_CURATED_DIR / (filename + "_qa_pairs.json")
                    output_path = DEFAULT_FINAL_DIR / (filename + "_qa_pairs.json")
                    logger.info(f"Executing QA Pairs Saving for file: {save_as_input}")
                    save_process = convert_format(
                        input_path=save_as_input,
                        output_path=output_path,
                        format_type="ft",
                        storage_format="json",
                        config_path=get_config_path(),
                        verbose=GLOBAL_DEBUG_FLAG,
                    )

                    logger.info("All files processed successfully")
                except Exception as e:
                    logger.error(f"Error processing files: {e}")
        except Exception as e:
            logger.error(f"Error processing files: {e}")
        finally:
            task_running.clear()
            task_complete.set()

    try:
        task_files = request.get_json()
        logger.info(f"Starting to process {len(task_files)} files")
        task_running.set()
        task_complete.clear()  # Reset completion flag

        # Start background thread
        thread = threading.Thread(target=process_files, args=(task_files,))
        thread.daemon = True
        thread.start()

        return jsonify({"status": "started", "message": "Task processing started"})
    except Exception as e:
        logger.error(f"Error starting task: {e}")
        task_running.clear()
        return jsonify({"error": str(e)}), 500


@app.route("/stream_task_log", methods=["GET"])
def stream_task_log():
    # Create a new queue for this client
    log_queue = queue.Queue()
    add_sse_queue(log_queue)

    def generate():
        try:
            while True:
                # Check if task is complete or client disconnected
                if task_complete.is_set():
                    # yield f"data: {GLBOAL_TASK_COMPLETED_MESSAGE}"
                    yield 'data: ' + GLBOAL_TASK_COMPLETED_MESSAGE+ '\n\n'
                    break

                try:
                    # Short timeout to frequently check task completion
                    message = log_queue.get(timeout=10)
                    yield f"data: {message}\n\n"
                except queue.Empty:
                    # Check completion again after timeout
                    if task_complete.is_set():
                        yield 'data: ' + GLBOAL_TASK_COMPLETED_MESSAGE+ '\n\n'
                        break
                    # Send keep-alive
                    yield ": keep-alive\n\n"
        finally:
            # Always clean up the queue
            remove_sse_queue(log_queue)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/set_config", methods=["GET", "POST"])
def set_config():
    """Upload a file as system configuration"""
    form = ConfigForm()

    if form.validate_on_submit():
        filename = "config.yaml"
        filepath = DEFAULT_CONFIG_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            content = request.form.get("editor")
            content = content.replace("\r\n", "\n")
            f.write(content)
        flash(f"File updated successfully: {filename}", "success")
        # Reload the configuration after saving the file
        reload_config()
        return redirect(url_for("index"))

    return render_template("config.html", form=form)


@app.route("/create", methods=["GET", "POST"])
def create():
    """Create content from text"""
    form = CreateForm()
    provider = get_llm_provider(GLOBAL_CONFIG)

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            content_type = form.content_type.data
            num_pairs = form.num_pairs.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            output_path = process_file(
                file_path=input_file,
                output_dir=str(DEFAULT_GENERATED_DIR),
                content_type=content_type,
                num_pairs=num_pairs,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
            )

            content_type_labels = {
                "qa": "QA pairs",
                "summary": "summary",
                "cot": "Chain of Thought examples",
                "cot-enhance": "CoT enhanced conversation",
            }
            content_label = content_type_labels.get(content_type, content_type)

            flash(
                f"Successfully generated {content_label}! Output saved to: {output_path}", "success"
            )
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available input files
    input_files = []
    if DEFAULT_OUTPUT_DIR.exists():
        input_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_OUTPUT_DIR.glob("*.txt")
        ]

    return render_template("create.html", form=form, provider=provider, input_files=input_files)


@app.route("/curate", methods=["GET", "POST"])
def curate():
    """Curate QA pairs interface"""
    form = CurateForm()
    provider = get_llm_provider(GLOBAL_CONFIG)

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            num_pairs = form.num_pairs.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            # Create output path
            filename = Path(input_file).stem
            output_file = f"{filename}_cleaned.json"
            output_path = DEFAULT_CURATED_DIR / output_file

            result_path = curate_qa_pairs(
                input_path=input_file,
                output_path=output_path,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
            )

            flash(f"Successfully curated QA pairs! Output saved to: {result_path}", "success")
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(result_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available JSON files
    json_files = []
    if DEFAULT_GENERATED_DIR.exists():
        json_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent))
            for f in DEFAULT_GENERATED_DIR.glob("*.json")
        ]

    return render_template("curate.html", form=form, provider=provider, json_files=json_files)


@app.route("/files")
def files():
    """File browser"""
    # Get all files in the data directory
    output_files = []
    generated_files = []

    if DEFAULT_OUTPUT_DIR.exists():
        output_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_OUTPUT_DIR.glob("*.*")
        ]

    if DEFAULT_GENERATED_DIR.exists():
        generated_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_GENERATED_DIR.glob("*.*")
        ]

    if DEFAULT_CURATED_DIR.exists():
        curated_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_CURATED_DIR.glob("*.*")
        ]

    if DEFAULT_FINAL_DIR.exists():
        # sft_final_files = [
        #     str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_FINAL_DIR.glob("*.*")
        # ]
        sft_final_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent))
            for f in DEFAULT_FINAL_DIR.iterdir()  # iterdir() lists all directory contents
        ]

    return render_template(
        "files.html",
        output_files=output_files,
        generated_files=generated_files,
        curated_files=curated_files,
        sft_final_files=sft_final_files,
    )


@app.route("/view/<path:file_path>")
def view_file(file_path):
    """View a file's contents"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists():
        flash(f"File not found: {file_path}", "danger")
        return redirect(url_for("files"))

    file_content = None
    file_type = "text"

    if full_path.suffix.lower() == ".json":
        try:
            with open(full_path, "r") as f:
                file_content = json.load(f)
            file_type = "json"

            # Detect specific JSON formats
            is_qa_pairs = "qa_pairs" in file_content
            is_cot_examples = "cot_examples" in file_content
            has_conversations = "conversations" in file_content
            has_summary = "summary" in file_content

        except Exception as e:
            # If JSON parsing fails, treat as text
            with open(full_path, "r") as f:
                file_content = f.read()
            file_type = "text"
            is_qa_pairs = False
            is_cot_examples = False
            has_conversations = False
            has_summary = False
    elif full_path.suffix.lower() == "":
        # it is a directory
        file_content = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent))
            for f in Path(full_path).iterdir()  # iterdir() lists all directory contents
        ]
        file_type = "directory"
        is_qa_pairs = False
        is_cot_examples = False
        has_conversations = False
        has_summary = False
    else:
        # Read as text
        with open(full_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        file_type = "text"
        is_qa_pairs = False
        is_cot_examples = False
        has_conversations = False
        has_summary = False

    return render_template(
        "view_file.html",
        file_path=file_path,
        file_type=file_type,
        content=file_content,
        is_qa_pairs=is_qa_pairs,
        is_cot_examples=is_cot_examples,
        has_conversations=has_conversations,
        has_summary=has_summary,
    )


@app.route("/ingest", methods=["GET", "POST"])
def ingest():
    """Ingest and parse documents"""
    form = IngestForm()

    if form.validate_on_submit():
        try:
            input_type = form.input_type.data
            output_name = form.output_name.data or None

            # Get default output directory for parsed files
            output_dir = str(DEFAULT_OUTPUT_DIR)

            if input_type == "file":
                # Handle file upload
                if not form.upload_file.data:
                    flash("Please upload a file", "warning")
                    return render_template("ingest.html", form=form)

                # Save the uploaded file to a temporary location
                temp_file = form.upload_file.data
                original_filename = temp_file.filename
                file_extension = Path(original_filename).suffix

                # Use upload filename as the output name if not provided
                if not output_name:
                    output_name = Path(original_filename).stem

                # Create a temporary file path in the output directory
                temp_path = DEFAULT_OUTPUT_DIR / f"temp_{output_name}{file_extension}"
                temp_file.save(temp_path)

                # Process the file
                input_path = str(temp_path)
            else:
                # URL or local path
                input_path = form.input_path.data
                if not input_path:
                    flash("Please enter a valid path or URL", "warning")
                    return render_template("ingest.html", form=form)

            # Process the file or URL
            output_path = ingest_process_file(
                file_path=input_path,
                output_dir=output_dir,
                output_name=output_name,
                config=GLOBAL_CONFIG,
            )

            # Clean up temporary file if it was an upload
            if input_type == "file" and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

            flash(f"Successfully parsed document! Output saved to: {output_path}", "success")
            return redirect(
                url_for(
                    "view_file",
                    file_path=str(Path(output_path).relative_to(DEFAULT_DATA_DIR.parent)),
                )
            )

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get some example URLs for different document types
    examples = {
        "PDF": "path/to/document.pdf",
        "YouTube": "https://www.youtube.com/watch?v=example",
        "Web Page": "https://example.com/article",
        "Word Document": "path/to/document.docx",
        "PowerPoint": "path/to/presentation.pptx",
        "Text File": "path/to/document.txt",
    }

    return render_template("ingest.html", form=form, examples=examples)


@app.route("/save", methods=["GET", "POST"])
def save():
    """Ingest and parse documents"""
    form = SaveAsForm()
    provider = get_llm_provider(GLOBAL_CONFIG)

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            output_format = form.output_format.data
            storage_format = form.storage_format.data

            # Create output path
            filename = str(Path(input_file).resolve())
            output_file = f"{Path(input_file).stem}.{storage_format}"
            output_path = str(Path(DEFAULT_FINAL_DIR) / output_file)

            result_path = convert_format(
                input_path=filename,
                output_path=output_path,
                format_type=output_format,
                storage_format=storage_format,
            )

            flash(f"Successfully generated SFT dataset! Output saved to: {result_path}", "success")
            return redirect(url_for("files"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available JSON files
    json_files = []
    if DEFAULT_GENERATED_DIR.exists():
        json_files = [
            str(f.relative_to(DEFAULT_DATA_DIR.parent))
            for f in DEFAULT_GENERATED_DIR.glob("*.json")
        ]
        json_files.extend(
            str(f.relative_to(DEFAULT_DATA_DIR.parent)) for f in DEFAULT_CURATED_DIR.glob("*.json")
        )

    return render_template("save.html", form=form, provider=provider, json_files=json_files)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload a file to the data directory"""
    form = UploadForm()

    if form.validate_on_submit():
        f = form.file.data
        filename = f.filename
        filepath = DEFAULT_OUTPUT_DIR / filename
        f.save(filepath)
        flash(f"File uploaded successfully: {filename}", "success")
        return redirect(url_for("files"))

    return render_template("upload.html", form=form)


@app.route("/api/qa_json/<path:file_path>")
def qa_json(file_path):
    """Return QA pairs as JSON for the JSON viewer"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        abort(404)

    try:
        with open(full_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        abort(500)


@app.route("/api/edit_item/<path:file_path>", methods=["POST"])
def edit_item(file_path):
    """Edit an item in a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404

    try:
        # Get the request data
        data = request.json
        item_type = data.get("item_type")  # qa_pairs, cot_examples, conversations
        item_index = data.get("item_index")
        item_content = data.get("item_content")

        if not all([item_type, item_index is not None, item_content]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400

        # Read the file
        with open(full_path, "r") as f:
            file_content = json.load(f)

        # Update the item
        if item_type == "qa_pairs" and "qa_pairs" in file_content:
            if 0 <= item_index < len(file_content["qa_pairs"]):
                file_content["qa_pairs"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "cot_examples" and "cot_examples" in file_content:
            if 0 <= item_index < len(file_content["cot_examples"]):
                file_content["cot_examples"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "conversations" and "conversations" in file_content:
            if 0 <= item_index < len(file_content["conversations"]):
                file_content["conversations"][item_index] = item_content
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400

        # Write back to the file
        with open(full_path, "w") as f:
            json.dump(file_content, f, indent=2)

        return jsonify({"success": True, "message": "Item updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/delete_item/<path:file_path>", methods=["POST"])
def delete_item(file_path):
    """Delete an item from a JSON file"""
    full_path = Path(DEFAULT_DATA_DIR.parent, file_path)

    if not full_path.exists() or full_path.suffix.lower() != ".json":
        return jsonify({"success": False, "message": "File not found or not a JSON file"}), 404

    try:
        # Get the request data
        data = request.json
        item_type = data.get("item_type")  # qa_pairs, cot_examples, conversations
        item_index = data.get("item_index")

        if not all([item_type, item_index is not None]):
            return jsonify({"success": False, "message": "Missing required parameters"}), 400

        # Read the file
        with open(full_path, "r") as f:
            file_content = json.load(f)

        # Delete the item
        if item_type == "qa_pairs" and "qa_pairs" in file_content:
            if 0 <= item_index < len(file_content["qa_pairs"]):
                file_content["qa_pairs"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "cot_examples" and "cot_examples" in file_content:
            if 0 <= item_index < len(file_content["cot_examples"]):
                file_content["cot_examples"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        elif item_type == "conversations" and "conversations" in file_content:
            if 0 <= item_index < len(file_content["conversations"]):
                file_content["conversations"].pop(item_index)
            else:
                return jsonify({"success": False, "message": "Invalid item index"}), 400
        else:
            return jsonify({"success": False, "message": "Invalid item type"}), 400

        # Write back to the file
        with open(full_path, "w") as f:
            json.dump(file_content, f, indent=2)

        return jsonify({"success": True, "message": "Item deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask server"""
    global GLOBAL_DEBUG_FLAG
    GLOBAL_DEBUG_FLAG = debug
    logger.info(f"Run the Flask server: {host}:{port}")
    logger.info(f"Mode: {("debug" if debug else "production") }")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
