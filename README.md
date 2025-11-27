# 🚀 CSV Explorer – Installation & Usage Guide

CSV Explorer is a lightweight Streamlit application that allows you to quickly visualize and analyze CSV files. Everything is automated: simply run the launcher and the environment will configure itself.

---

# 📦 Requirements

Before running the tool, please ensure the following requirements are met.

## ✅ 1. **Python Version**

CSV Explorer requires:

* **Python 3.10, 3.11, or 3.12 (64-bit only)**
* Installed from the official site:
  👉 [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

⚠️ **32-bit Python is NOT supported.**
Pandas and other scientific libraries do not ship 32-bit Windows wheels — pip will try (and fail) to compile them, causing Meson / VS Build Tools errors.

### How to verify your Python installation:

Open a command prompt and run:

```bat
py --version
py -c "import platform; print(platform.architecture())"
```

Expected output:

```
Python 3.11.x
('64bit', 'WindowsPE')
```

If you see **32bit**, uninstall Python and install the 64-bit version.

---

## ✅ 2. **Windows OS**

CSV Explorer is tested on:

* **Windows 10**
* **Windows 11**

macOS & Linux require launching via terminal (no `.bat` script provided yet).

---

## ✅ 3. **No admin rights required**

Installation works entirely in the project folder:

* A local virtual environment is created (`.venv`)
* Required Python packages are installed locally
* No system changes are made

---

# ▶️ How to Run the Application

### 1. **Download or clone the repository**

```
git clone https://github.com/fkohler-hydrique/csv-explorer
cd csv-explorer
```

### 2. **Double-click the launcher**

```
run_app.bat
```

That’s it.

The launcher will:

1. Check your Python installation (version + architecture)
2. Create a virtual environment (`.venv`)
3. Upgrade pip
4. Install all required dependencies (Streamlit, pandas, plotly, etc.)
5. Launch the CSV Explorer web app in your browser

You’ll see status messages like:

```
[SETUP] Step 1/3: Creating virtual environment
[SETUP] Step 2/3: Upgrading pip
[SETUP] Step 3/3: Installing required packages
[INFO] Setup complete. Starting CSV Explorer app...
```

Next, your browser will open at:

📍 **[http://localhost:8501](http://localhost:8501)**

---

# ❗ Common Issues & Solutions

## 🔴 **"Python version is too old" (3.8 or below)**

Install Python 3.10–3.12 (64-bit):
[https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

---

## 🔴 **"Python version is newer than tested" (3.14+)**

The launcher prevents untested versions from being used.

➡️ Install Python 3.10–3.12 instead.

---

## 🔴 **"32-bit Python detected"**

Your Python installation is incompatible.

Check architecture:

```bat
py -c "import platform; print(platform.architecture())"
```

If you see `"32bit"`:

1. Uninstall Python
2. Download **Windows 64-bit installer** from python.org
3. Re-run the `.bat` launcher

---

## 🔴 pip install fails with errors mentioning:

* **Meson**
* **vswhere.exe**
* **Microsoft Visual C++ Build Tools**

This means Python is 32-bit or incorrectly installed.

➡️ Fix: Use a clean **64-bit Python from python.org**
➡️ Use Python 3.10–3.12 (recommended)

---

## 🔴 Corporate proxy prevents pip from downloading packages

If you're behind a proxy:

```bat
pip config set global.proxy http://username:password@proxy:port
```

Or consult your IT team.

---

# 🧪 Troubleshooting

Run:

```
troubleshoot_python.bat
```

This script prints:

* Python version
* Architecture
* Pip version
* Interpreter paths

You can share this output with support if needed.

---

# 🗑️ Resetting the environment (if something breaks)

You can safely delete:

```
.venv/
.deps_installed
pip_install.log
pip_upgrade.log
```

Then re-run:

```
run_app.bat
```

A fresh environment will be created.

---

# 📁 Project Structure

```
csv-explorer/
│   run_app.bat
│   troubleshoot_python.bat
│   requirements.txt
│   README.md
│   app.py
│
└─── .venv/              (auto-created virtual environment)
```

---

# 🎉 You’re Ready to Use CSV Explorer!

Once launched, you can:

* Upload CSV files
* View data tables
* Explore charts (Plotly)
* Generate insights
