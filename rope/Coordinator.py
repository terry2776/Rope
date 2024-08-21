import threading
import time
import torch
from torchvision import transforms
from queue import Queue, Empty
import inspect

import rope.GUI as GUI
import rope.VideoManager as VM
import rope.Models as Models
from rope.external.clipseg import CLIPDensePredT

class AppState:
    def __init__(self):
        self.frame = []
        self.r_frame = []
        self.resize_delay = 1
        self.mem_delay = 1
        self.stop_event = threading.Event()
        self.main_thread = None
        self.busy = False  # Flag to indicate if the system is processing a critical action

    def start_thread(self, target):
        self.main_thread = threading.Thread(target=target)
        self.main_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.main_thread:
            self.main_thread.join()

def handle_action(action, vm, state):
    handlers = {
        "load_target_video": vm.load_target_video,
        "load_target_image": vm.load_target_image,
        "play_video": vm.play_video,
        "get_requested_video_frame": lambda value: get_requested_video_frame_sync(value, vm, state),
        "get_requested_video_frame_without_markers": lambda value: get_requested_video_frame_sync(value, vm, state, marker=False),
        "get_requested_frame": vm.get_requested_frame,
        "enable_virtualcam": vm.enable_virtualcam,
        "disable_virtualcam": vm.disable_virtualcam,
        "change_webcam_resolution_and_fps": vm.change_webcam_resolution_and_fps,
        "target_faces": vm.assign_found_faces,
        "saved_video_path": lambda value: setattr(vm, 'saved_video_path', value),
        "vid_qual": lambda value: setattr(vm, 'vid_qual', int(value)),
        "set_stop": lambda value: setattr(vm, 'stop_marker', value),
        "perf_test": lambda value: setattr(vm, 'perf_test', value),
        "ui_vars": lambda value: setattr(vm, 'ui_data', value),
        "control": lambda value: setattr(vm, 'control', value),
        "parameters": lambda value: handle_parameters(value, vm),
        "markers": lambda value: setattr(vm, 'markers', value),
        "clear_mem": vm.clear_mem,
        "stop_play": gui.set_player_buttons_to_inactive,
        "set_virtual_cam_toggle_disable": gui.set_virtual_cam_toggle_disable,
        "disable_record_button": gui.disable_record_button,
        "clear_faces_stop_swap": lambda _: (gui.clear_faces(), gui.toggle_swapper(0)),
        "clear_stop_enhance": lambda _: gui.toggle_enhancer(0),
        "set_slider_length": gui.set_video_slider_length,
        "update_markers_canvas": gui.update_markers_canvas,
        "face_landmarks": lambda value: setattr(vm, 'face_landmarks', value),
        "function": lambda func: eval(func),
    }

    if action[0] in handlers:
        handler = handlers[action[0]]

        # Check how many arguments the handler expects
        if len(inspect.signature(handler).parameters) == 0:
            handler()  # Call the function without arguments
        else:
            handler(action[1])  # Call the function with one argument

        return True
    else:
        print(f"Action not found: {action[0]} {action[1]}")
        return False

def get_requested_video_frame_sync(value, vm, state, marker=True):
    state.busy = True  # Set the busy flag
    gui.config(cursor="watch")
    try:
        vm.get_requested_video_frame(value, marker=marker)
    except Exception as e:
        print(f"Error during get_requested_video_frame: {e}")
    finally:
        gui.config(cursor="")
        state.busy = False  # Reset the busy flag after completion

def handle_parameters(params, vm):
    if params["CLIPSwitch"] and not vm.clip_session:
        vm.clip_session = load_clip_model()
    vm.parameters = params

def loop(state, action_queue):
    while not state.stop_event.is_set():
        # Perform periodic tasks regardless of busy state
        if state.mem_delay > 1000:
            gui.update_vram_indicator()  # Update the VRAM indicator periodically
            state.mem_delay = 0
        else:
            state.mem_delay += 1

        # Process actions only if not busy
        if not state.busy:
            try:
                action = action_queue.get(timeout=0.001)
                if handle_action(action, vm, state):
                    action_queue.task_done()
            except Empty:
                continue

        time.sleep(0.01)  # Add a small sleep to avoid busy-waiting and reduce CPU usage

def quit_app(state):
    state.stop()
    gui.destroy()

def coordinator_gui(state, action_queue):
    if vm.get_frame_length() > 0:
        state.frame.append(vm.get_frame())

    if len(state.frame) > 0:
        gui.set_image(state.frame[0], False)
        state.frame.pop(0)

    if vm.get_requested_frame_length() > 0:
        state.r_frame.append(vm.get_requested_frame())
    if len(state.r_frame) > 0:
        gui.set_image(state.r_frame[0], True)
        state.r_frame = []

    if gui.get_action_length() > 0:
        action_queue.put(gui.get_action())
    if vm.get_action_length() > 0:
        action_queue.put(vm.get_action())

    vm.process()
    gui.after(10, lambda: coordinator_gui(state, action_queue))  # Reschedule the GUI update

def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    clip_session.eval()
    clip_session.load_state_dict(torch.load('./models/rd64-uni-refined.pth'), strict=False)
    clip_session.to(device)
    return clip_session

def run():
    global gui, vm, state
    state = AppState()
    models = Models.Models()
    gui = GUI.GUI(models)
    vm = VM.VideoManager(models)

    gui.initialize_gui()
    gui.protocol("WM_DELETE_WINDOW", lambda: quit_app(state))

    action_queue = Queue()

    state.start_thread(target=lambda: loop(state, action_queue))

    gui.after(10, lambda: coordinator_gui(state, action_queue))
    gui.mainloop()