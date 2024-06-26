/**
 * @file main.cpp
 * @author Jakub Ochnik (jakub.ochnik1@gmail.com)
 * @name Cube LUT loader
 * @version 0.1
 * @date 2022-03-27
 *
 * @license MIT
 *
 */
#include <cli/cli.hpp>
#include <TaskDispatcher/TaskDispatcher.hpp>

int main(int argc, char* argv[])
{	
	TaskDispatcher programEntry(argc, argv);
	return programEntry.start();
}
