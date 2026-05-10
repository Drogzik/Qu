#define MyAppName "Pomoshnik"
#define MyAppVersion "1.0.0"
; Absolute paths from MSBuild (Release x64): ISCC /DBuildDir=... /DUiDir=...
#ifndef BuildDir
#define BuildDir "..\ConsoleApplication1\ConsoleApplication1\x64\Release"
#endif
#ifndef UiDir
#define UiDir "..\ConsoleApplication1\ConsoleApplication1\ui"
#endif

[Setup]
AppId={{8F3E2B1A-9C0D-4E5F-A7B6-1D2E3F4A5B6C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppName}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=Output
OutputBaseFilename=PomoshnikSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\Pomoshnik.exe

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Dirs]
Name: "{app}\models"; Flags: uninsalwaysuninstall

[Files]
Source: "{#BuildDir}\Pomoshnik.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildDir}\*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#UiDir}\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildDir}\models\readme.txt"; DestDir: "{app}\models"; Flags: ignoreversion skipifsourcedoesntexist
Source: "download_model.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\Pomoshnik.exe"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  PsLine: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Skachivanie modeli II (~2 GB, nuzhen internet)...';
    WizardForm.Update;
    PsLine := ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\download_model.ps1" "{app}"');
    if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), PsLine, '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) then
    begin
      if ResultCode <> 0 then
        MsgBox('Avtozagruzka modeli ne udalas (prover internet). Polozhi fayl .gguf v papku models vruchnuyu.', mbInformation, MB_OK);
    end
    else
      MsgBox('Ne udalos zapustit zagruzchik modeli.', mbInformation, MB_OK);
  end;
end;