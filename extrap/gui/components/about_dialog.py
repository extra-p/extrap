import itertools
import logging
from urllib.error import HTTPError, URLError

from PySide6 import QtWidgets
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QLayout, QLabel, QSpacerItem, QStyle, QCheckBox, QDialogButtonBox

import extrap

_SETTING_CHECK_FOR_UPDATES_ON_STARTUP = 'check_for_updates_on_startup'


class AboutDialog(QtWidgets.QDialog):

    def __init__(self, main_widget):
        super(AboutDialog, self).__init__(main_widget)
        self.main_widget = main_widget
        self._init_ui()
        self.setWindowTitle("About " + extrap.__title__)

    def _init_ui(self):
        layout = QGridLayout()
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.setSpacing(4)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 1)

        row = itertools.count(0)

        columnSpan = 4
        layout.addWidget(QLabel(f"<h1>{extrap.__title__}</h1>"), next(row), 0, 1, columnSpan)
        layout.addWidget(QLabel(f"<b>{extrap.__description__}</b>"), next(row), 0, 1, columnSpan)
        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)

        icon_label = QLabel()
        text_label = QLabel()
        text_label.setOpenExternalLinks(True)
        same_row = next(row)
        layout.addWidget(icon_label, same_row, 0, 1, 1)
        layout.addWidget(text_label, same_row, 1, 1, columnSpan - 1)
        try:
            update_available = self.update_available()
            if not update_available:
                icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton).pixmap(16))
                text_label.setText(f"{extrap.__title__} is up to date")
            else:
                icon_label.setPixmap(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation).pixmap(16))
                text_label.setText(f'Version {update_available[0]} is available. '
                                   f'Get it here: <a href="{update_available[1]}">{update_available[1]}</a>')
        except HTTPError as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e))
        except URLError as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e.reason))
        except Exception as e:
            icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16))
            text_label.setText(f"Could not check for updates: " + str(e))

        same_row = next(row)
        layout.addWidget(QLabel(f"Version {extrap.__version__}"), same_row, 1, 1, 1)
        layout.addWidget(QLabel(' — '), same_row, 2, 1, 1)

        check_for_updates = QCheckBox(self)
        check_for_updates.setChecked(
            bool(self.main_widget.settings.value(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, True, bool)))
        check_for_updates.setText("Check for updates on start up")
        check_for_updates.toggled.connect(
            lambda status: self.main_widget.settings.setValue(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, status))
        layout.addWidget(check_for_updates, same_row, 3, 1, 1)

        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)

        creators = QLabel(extrap.__developed_by_html__)
        creators.setOpenExternalLinks(True)
        layout.addWidget(creators, next(row), 0, 1, columnSpan)
        layout.addItem(QSpacerItem(0, 2), next(row), 0, 1, columnSpan)
        support = QLabel(f'Do you have questions or suggestions?<br>'
                         f'Write us: <a href="mailto:{extrap.__support_email__}">{extrap.__support_email__}</a>')
        support.setOpenExternalLinks(True)
        layout.addWidget(support, next(row), 0, 1, columnSpan)

        layout.addItem(QSpacerItem(0, 10), next(row), 0, 1, columnSpan)

        layout.addWidget(QLabel(extrap.__copyright__), next(row), 0, 1, columnSpan)

        layout.addItem(QSpacerItem(0, 10), next(row), 0, 1, columnSpan)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)

        # button_box.addButton(check_for_updates, QDialogButtonBox.ButtonRole.ResetRole)
        # ResetRole is a hack to achieve left alignment
        layout.addWidget(button_box, next(row), 0, 1, columnSpan)
        self.setLayout(layout)

    @staticmethod
    def update_available():
        import json
        import urllib.request
        from packaging.version import Version

        with urllib.request.urlopen(extrap.__current_version_api__) as response:
            data = json.loads(response.read().decode('utf-8'))
            info = data['info']

            if Version(info['version']) > Version(extrap.__version__):
                return info['version'], info['release_url']
            else:
                return False

    @classmethod
    def check_updates_add_notification(cls, settings, menubar):
        if settings.value(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP, True, bool):
            update_available = None
            try:
                update_available = cls.update_available()
            except Exception as e:
                logging.error("Check for updates: " + str(e))

            if update_available:
                update_menu = menubar.addMenu("UPDATE AVAILABLE")
                update_action = update_menu.addAction(
                    f'Version {update_available[0]} is available here: {update_available[1]}'
                )
                update_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(update_available[1])))

                ignore_action = update_menu.addAction("Ignore")
                ignore_action.triggered.connect(lambda: update_menu.menuAction().setVisible(False))
                update_menu.addSeparator()
                auto_update_toggle = update_menu.addAction('Check for updates on startup')
                auto_update_toggle.setChecked(True)
                auto_update_toggle.setCheckable(True)
                update_menu.addAction(auto_update_toggle)
                auto_update_toggle.toggled.connect(
                    lambda toggled: settings.setValue(_SETTING_CHECK_FOR_UPDATES_ON_STARTUP,
                                                      toggled))
