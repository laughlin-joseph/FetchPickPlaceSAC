import os
import subprocess
from tensorboard import program
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

class LauncherGUI(BoxLayout):
    def __init__(self, **kwargs):
        super(LauncherGUI, self).__init__(**kwargs)

class LoadDialog(FloatLayout):
    def __init__(self, **kwargs):
        super(LoadDialog, self).__init__(**kwargs)
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class TensorboardApp(App):
    def __init__(self, **kwargs):
        super(TensorboardApp, self).__init__(**kwargs)
        self.tensorboard_proc = None
        self.tensorboard_logfile_dir = ''

    def build(self):
        return LauncherGUI()
    
    def dismiss_popup(self, caller=None):
        self._popup.dismiss()

    def load(self, path, selection):
        self.root.ids['directory_input'].text = path
        self.tensorboard_logfile_dir = path
        self.dismiss_popup()

    def select_directory(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def show_message(self, message):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=message))
        content.add_widget(Button(text='Close', on_press=self.dismiss_popup))
        self._popup = Popup(title="Info", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def toggle_tensorboard(self, caller):
        dir_path = self.tensorboard_logfile_dir if self.tensorboard_logfile_dir else self.root.ids['directory_input'].text
        if not dir_path:
            self.show_message('Error: Tensorboard logfile directory not specified.')
            return
        if not os.path.exists(dir_path):
            self.show_message('Error: Tensorboard logfile directory does not exist.')
            return
        if self.tensorboard_proc is None:
            try:
                self.tensorboard_proc = subprocess.Popen(['python','-m','tensorboard.main','--logdir=%s'%dir_path])
            except Exception as e:
                self.tensorboard_proc = None
                print('\nError starting Tensorboard: %s' % e)
            caller.text = 'Stop TensorBoard'
            caller.background_color = (0.7, 0.3, 0.5, 1)
            self.show_message('TensorBoard started.')
        else:
            self.tensorboard_proc.terminate()
            self.tensorboard_proc = None
            caller.text = 'Start TensorBoard'
            caller.background_color = (0.3, 0.7, 0.5, 1)
            self.show_message('TensorBoard stopped.')
        
if __name__ == '__main__':
    TensorboardApp().run()