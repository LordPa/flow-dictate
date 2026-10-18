' Flow Dictate: starts the tray icon with no visible window. Double-click to run.
Set sh = CreateObject("WScript.Shell")
dir = Replace(WScript.ScriptFullName, WScript.ScriptName, "")
sh.CurrentDirectory = dir
sh.Run """" & dir & "start.bat"" tray", 0, False
