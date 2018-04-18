function os.winSdkVersion()
    local reg_arch = iif( os.is64bit(), "\\Wow6432Node\\", "\\" )
    local sdk_version = os.getWindowsRegistry( "HKLM:SOFTWARE" .. reg_arch .."Microsoft\\Microsoft SDKs\\Windows\\v10.0\\ProductVersion" )
    if sdk_version ~= nil then return sdk_version end
end

workspace "ai-zero"
	location ( "_build/%{_ACTION}" )
	architecture "x64"
	configurations { "Debug", "Release" }

	configuration "vs*"
		defines { "_CRT_SECURE_NO_WARNINGS" }	

	filter "configurations:Debug"
		targetdir ( "_build/%{_ACTION}/bin/Debug" )
	 	defines { "DEBUG" }
		symbols "On"

	filter "configurations:Release"
		targetdir ( "_build/%{_ACTION}/bin/Release" )
		defines { "NDEBUG" }
		optimize "On"

	-- vs2017 windows sdk problem workaround:
	filter {"system:windows", "action:vs*"}
    	systemversion(os.winSdkVersion() .. ".0")

    exceptionhandling("off")

    filter {}

project "gradient-check"
	kind "ConsoleApp"
	language "C++"
	files { "examples/gradient-check/**.cpp" }
	includedirs { "include" }

project "mnist-cnn"
	kind "ConsoleApp"
	language "C++"
	files { "examples/gradient-check/**.cpp" }
	includedirs { "include" }

project "rl-connect4"
	kind "ConsoleApp"
	language "C++"
	files { "examples/gradient-check/**.cpp" }
	includedirs { "include" }
