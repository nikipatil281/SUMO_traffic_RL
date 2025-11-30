# scripts/check_env.py
import os, sys, platform, json, shutil, subprocess, textwrap

def which_exe(candidates):
    for c in candidates:
        p = shutil.which(c)
        if p: return p
    return None

def try_run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "stdout": out.strip()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

summary = {
    "python": {
        "executable": sys.executable,
        "version": sys.version.replace("\n"," "),
        "platform": platform.platform(),
    },
    "pip_version": try_run([sys.executable, "-m", "pip", "--version"]),
    "env": {
        "SUMO_HOME": os.environ.get("SUMO_HOME"),
        "PATH_has_sumo": bool(which_exe(["sumo"] + (["sumo.exe"] if os.name=="nt" else []))),
        "PATH_has_sumo_gui": bool(which_exe(["sumo-gui"] + (["sumo-gui.exe"] if os.name=="nt" else []))),
    },
    "sumo": {},
    "python_pkgs": {}
}

# Detect SUMO binaries
sumo_bin = which_exe(["sumo", "sumo.exe"])
sumo_gui_bin = which_exe(["sumo-gui", "sumo-gui.exe"])

# Fallback to SUMO_HOME/bin if needed
sh = os.environ.get("SUMO_HOME")
if not sumo_bin and sh:
    guess = os.path.join(sh, "bin", "sumo.exe" if os.name=="nt" else "sumo")
    if os.path.exists(guess): sumo_bin = guess
if not sumo_gui_bin and sh:
    guess = os.path.join(sh, "bin", "sumo-gui.exe" if os.name=="nt" else "sumo-gui")
    if os.path.exists(guess): sumo_gui_bin = guess

# Query SUMO versions (CLI, reliable on all setups)
summary["sumo"]["sumo_path"] = sumo_bin
summary["sumo"]["sumo_gui_path"] = sumo_gui_bin
if sumo_bin:
    summary["sumo"]["sumo_cli_version"] = try_run([sumo_bin, "--version"])
if sumo_gui_bin:
    summary["sumo"]["sumo_gui_version"] = try_run([sumo_gui_bin, "--version"])

# Try importing SUMO Python APIs
def try_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", None)
        return {"import_ok": True, "version_attr": ver}
    except Exception as e:
        return {"import_ok": False, "error": f"{type(e).__name__}: {e}"}

summary["python_pkgs"]["sumolib"] = try_import("sumolib")
summary["python_pkgs"]["traci"]   = try_import("traci")

# If traci import worked, show how we'd query TraCI server version (doc note only)
summary["notes"] = textwrap.dedent("""
  If needed later, TraCI can report the running SUMO server version via traci.getVersion()
  after a connection is opened. For now we only use CLI '--version' checks.
""").strip()

print(json.dumps(summary, indent=2))
