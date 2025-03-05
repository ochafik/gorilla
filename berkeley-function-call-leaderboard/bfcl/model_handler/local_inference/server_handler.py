import subprocess
import threading
import time

import requests


class LocalServerHandler:
    def __init__(self, command: list[str] | None, host: str, port: int, ready_path: str):
        self.ready_url = f"http://{host}:{port}${ready_path}"

        if command:
            self.process = subprocess.Popen(command,
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.PIPE,  # Capture stderr
                text=True,  # To get the output as text instead of bytes
            )
            self.skip_server_setup = False
        else:
            self.skip_server_setup = True

        self.stop_event = threading.Event()
        
        # Start threads to read and print stdout and stderr
        self.stdout_thread = threading.Thread(
            target=self.log_subprocess_output, args=(self.process.stdout, self.stop_event)
        )
        self.stderr_thread = threading.Thread(
            target=self.log_subprocess_output, args=(self.process.stderr, self.stop_event)
        )
        self.stdout_thread.start()
        self.stderr_thread.start()

    def wait_for_server_ready(self):
        # Wait for the server to be ready
        server_ready = False
        while not server_ready:
            # Check if the process has terminated unexpectedly
            if not self.skip_server_setup and self.process.poll() is not None:
                # Output the captured logs
                stdout, stderr = self.process.communicate()
                print(stdout)
                print(stderr)
                raise Exception(
                    f"Subprocess terminated unexpectedly with code {self.process.returncode}"
                )
            try:
                # Make a simple request to check if the server is up
                response = requests.get(self.ready_url)
                if response.status_code == 200:
                    server_ready = True
                    print("server is ready!")
            except requests.exceptions.ConnectionError:
                # If the connection is not ready, wait and try again
                time.sleep(1)

        if not self.skip_server_setup:
            # Signal threads to stop reading output
            self.stop_event.set()

        return server_ready
    
    def terminate(self):
        if not self.skip_server_setup:
            # Ensure the server process is terminated properly
            self.process.terminate()
            try:
                # Wait for the process to terminate fully
                self.process.wait(timeout=15)
                print("Process terminated successfully.")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()  # Wait again to ensure it's fully terminated
                print("Process killed.")

            # Wait for the output threads to finish
            self.stop_event.set()
            self.stdout_thread.join()
            self.stderr_thread.join()
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.stop_event = None

    def log_subprocess_output(pipe, stop_event):
        # Read lines until stop event is set
        for line in iter(pipe.readline, ""):
            if stop_event.is_set():
                break
            else:
                print(line, end="")
        pipe.close()
        print("server log tracking thread stopped successfully.")
