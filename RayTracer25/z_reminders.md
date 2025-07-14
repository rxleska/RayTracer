# Current Ideology Reminders
1. **Using code defined in headers**, instead of compiling code for each file, this will increase compilation time but will enhance the compiler's ability to optimize the code.
2. **Using float instead of double** cuda is 32 times faster with float than double (according to duck duck go assistant). 
3. **Using math.h instead of cmath** math.h is more efficient on the device than cmath, which works on the device but is not as efficient as it is on the host.
4. **Avoid using recursion** on the device, it is not efficient and can lead to stack overflow errors. Use iterative methods instead.
5. **Avoid using dynamic memory allocation** on the device, it leads to fragmentation and can cause performance issues. Use static memory allocation instead.
6. **Add the inline keyword**, even if it is already implicitly defined, to ensure that the compiler optimizes the code correctly and for better readability.
    - also look into **__forceinline** for functions that must be inlined.
7. **Avoid using getters and setters** on the device, they are not efficient and can lead to performance issues. Use direct access to the members instead.







### Recommendation for editing in vscode on a new device
1. **Install the C/C++ extension** for syntax highlighting and IntelliSense.
3. **add the following line to your settings.json file** to associate .cu files with the CUDA language:
   ```json
   "files.associations": {
        "*.cu": "cuda-cpp",
        "*.h": "cuda-cpp"
    }
   ```