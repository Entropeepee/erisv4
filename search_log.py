import re
log_path = r"C:\Users\david\Documents\Unreal Projects\ElderBoomHollowMassiveMed 5.8\Saved\Logs\ElderboomVillage.log"
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print("Searching for compiler errors...")
for i, line in enumerate(lines):
    if "Error:" in line and "Compiler" in line:
        print(f"Line {i}: {line.strip()}")
    elif "Compile error" in line or "KismetCompiler" in line and "Error" in line:
        print(f"Line {i}: {line.strip()}")
