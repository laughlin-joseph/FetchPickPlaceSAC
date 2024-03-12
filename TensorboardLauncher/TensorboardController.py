import os
import subprocess
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.widget import Widget

class LauncherGUI(Widget):
    pass

class TensorboardApp(App):
    def __init__(self, **kwargs):
        super(TensorboardApp, self).__init__(**kwargs)
        self.tensorboard_process = None

    def build(self):
        return LauncherGUI()
        
    def select_directory(self, instance):
        file_chooser = FileChooserIconView()
        file_chooser.bind(on_submit=self.directory_selected)
        file_chooser.popup()

    def directory_selected(self, instance):
        self.file_path_input.text = instance.selection[0]

    def toggle_tensorboard(self, instance):
        directory_path = self.file_path_input.text
        if not os.path.exists(directory_path):
            self.show_message('Error: Directory does not exist!')
            return

        if self.tensorboard_process is None:
            # Start TensorBoard
            command = ['tensorboard', '--logdir', directory_path]
            self.tensorboard_process = subprocess.Popen(command)
            self.tensorboard_button.text = 'Stop TensorBoard'
            self.show_message('TensorBoard started.')
        else:
            # Stop TensorBoard
            self.tensorboard_process.terminate()
            self.tensorboard_process = None
            self.tensorboard_button.text = 'Start TensorBoard'
            self.show_message('TensorBoard stopped.')

    def show_message(self, message):
        popup = BoxLayout(orientation='vertical', padding=10, spacing=10)
        popup.add_widget(Label(text=message))
        popup.add_widget(Button(text='Close', on_press=lambda *args: self.dismiss_popup()))
        self.popup = popup
        self.popup.open()

    def dismiss_popup(self):
        self.popup.dismiss()
        
if __name__ == '__main__':
    TensorboardApp().run()