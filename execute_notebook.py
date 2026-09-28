import nbformat as nbf
import io
import os
import sys
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

notebook_path = "AI_ML_Task2_Model_Comparison.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

exec_globals = {
    "__name__": "__main__"
}

execution_count = 1

for cell_idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        code = cell.source
        cell.outputs = []
        cell.execution_count = execution_count
        
        stdout_buf = io.StringIO()
        plt.close('all')
        
        captured_displays = []
        def custom_display(obj):
            if hasattr(obj, 'to_html'):
                captured_displays.append(("text/html", obj.to_html()))
            elif hasattr(obj, '_repr_html_'):
                captured_displays.append(("text/html", obj._repr_html_()))
            else:
                stdout_buf.write(str(obj) + "\n")
        
        exec_globals["display"] = custom_display
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                lines = code.strip().split("\n")
                last_line = lines[-1].strip() if lines else ""
                
                # Check if last line is an expression to display
                eval_keywords = ["import ", "from ", "for ", "while ", "if ", "def ", "class ", "print(", "plt.", "return", "#", "="]
                is_statement = any(last_line.startswith(kw) for kw in eval_keywords) or ("=" in last_line and not any(last_line.startswith(x) for x in ["==", "!=", "<=", ">="]))
                
                if not is_statement and len(last_line) > 0:
                    exec_code = "\n".join(lines[:-1])
                    if exec_code.strip():
                        exec(exec_code, exec_globals)
                    eval_res = eval(last_line, exec_globals)
                    if eval_res is not None:
                        if hasattr(eval_res, 'to_html'):
                            captured_displays.append(("text/html", eval_res.to_html()))
                        elif hasattr(eval_res, '_repr_html_'):
                            captured_displays.append(("text/html", eval_res._repr_html_()))
                        else:
                            stdout_buf.write(str(eval_res) + "\n")
                else:
                    exec(code, exec_globals)
        except Exception as e:
            stdout_buf.write(f"Error during cell execution: {e}\n")
            print(f"Cell {execution_count} (index {cell_idx}) failed: {e}")

        # Stream text output
        out_text = stdout_buf.getvalue()
        if out_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type="stream",
                name="stdout",
                text=out_text
            ))
            
        # Display data outputs (HTML / Rich tables)
        for mime, html_content in captured_displays:
            cell.outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={
                    mime: html_content,
                    "text/plain": "<DataFrame or Styler Object>"
                }
            ))
            
        # Matplotlib figures
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=140)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            cell.outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": img_b64,
                    "text/plain": "<Figure size ...>"
                }
            ))
        plt.close('all')
        
        execution_count += 1

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed all {execution_count - 1} code cells and populated outputs in {notebook_path}")
