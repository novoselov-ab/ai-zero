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

    filter {"system:macosx"}
        toolset "gcc"

    filter { "action:gmake" }
        buildoptions { "-std=c++17" }

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
