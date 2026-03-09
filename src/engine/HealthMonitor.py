'''
Health Monitor take game windows as input to calculate player's HP/MP/EXP Bar percentage.
When player's HP/MP drop to specific threshold, it'd press key to drink potion 
'''

# Standard Import
import threading
import time
import cv2

# Local Import
from src.utils.logger import logger
from src.utils.common import get_bar_percent
from src.input.KeyBoardController import press_key

class HealthMonitor:
    '''
    Independent health monitoring thread that can heal while other actions are running
    '''
    def __init__(self, cfg, kb_controller):
        self.cfg = cfg
        self.kb = kb_controller
        self.is_terminated = False
        self.enabled = True
        self.thread = None # health monitor thread

        # Health monitoring state
        self.hp_percent = 100
        self.mp_percent = 100
        self.exp_percent = 100

        # Timers
        self.t_last_heal = 0
        self.t_last_mp = 0
        self.t_last_hp_reduce = 0
        self.t_last_run = 0
        self.t_hp_watch_dog = time.time()
        self.t_last_bar_relocate = 0

        # Frame data (will be updated by main thread)
        self.img_frame = None
        self.frame_lock = threading.Lock()

        # FPS settings
        self.fps_limit = self.cfg["health_monitor"]["fps_limit"]
        self.bar_relocate_interval = self.cfg["health_monitor"].get(
            "bar_relocate_interval", 2.0
        )
        self.fps = 0

        # Debug information
        # hp/mp/exp bars loc and size, [(x,y,w,h), ...]
        self.loc_size_bars = [(0, 0, 0, 0),
                              (0, 0, 0, 0),
                              (0, 0, 0, 0)]

        logger.info("[Health Monitor] Init done")

    def start(self):
        '''
        Start health monitoring thread
        '''
        if not self.is_terminated:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("[Health Monitor] Started")

    def stop(self):
        '''
        Stop health monitoring thread
        '''
        self.is_terminated = True
        if self.thread:
            self.thread.join()
            logger.info("[Health Monitor] Terminated")

    def enable(self):
        '''
        Enable health monitoring
        '''
        self.enabled = True

    def disable(self):
        '''
        Disable health monitoring
        '''
        self.enabled = False

    def update_frame(self, img_frame):
        '''
        Update frame data from main thread
        '''
        with self.frame_lock:
            self.img_frame = img_frame

    def _is_valid_bar_regions(self, img_shape):
        '''
        Check whether cached bar rectangles are valid for current frame.
        '''
        h_img, w_img = img_shape[:2]
        for x, y, w, h in self.loc_size_bars:
            if w <= 0 or h <= 0:
                return False
            if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                return False
        return True

    def _get_percent_from_regions(self, img_frame, loc_size_bars):
        '''
        Compute bar percentages from cached or newly detected rectangles.
        '''
        percent_bars = []
        for x, y, w, h in loc_size_bars:
            percent_bars.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
        return percent_bars

    def _detect_bar_regions(self, img_frame):
        '''
        Detect HP/MP/EXP bar regions from current frame.
        '''
        img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(img_frame_gray, 240, 255)
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        loc_size_bars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            # for game window resolution 752x1282, w/h == 7.5, w*h == 3630
            if 4 < w/h < 10 and 2500 < w*h < 5000:
                loc_size_bars.append((x, y, w, h))

        # sort contours by x coordinate
        loc_size_bars = sorted(loc_size_bars, key=lambda bar: bar[0])
        if len(loc_size_bars) != 3:
            return None
        return loc_size_bars

    def get_hp_mp_exp_percent(self):
        '''
        Extracts the player's HP, MP, and EXP ratios from game frame.

        This function:
        - Crops the predefined HP, MP, and EXP bar regions from the game frame.
        - Identifies empty areas in each bar.
        - Computes the fill ratio for each bar as: 1 - (empty_pixels / total_pixels).

        Returns:
            tuple: (hp_percent, mp_percent, exp_percent), each a float between 0 and 1.
        '''
        if self.img_frame is None:
            return None, None, None

        with self.frame_lock:
            img_frame = self.img_frame.copy()

        has_cached_regions = self._is_valid_bar_regions(img_frame.shape)
        t_cur = time.time()
        need_relocate = (
            (not has_cached_regions) or
            (t_cur - self.t_last_bar_relocate >= self.bar_relocate_interval)
        )

        if need_relocate:
            loc_size_bars = self._detect_bar_regions(img_frame)
            if loc_size_bars is not None:
                self.loc_size_bars = loc_size_bars
                self.t_last_bar_relocate = t_cur
                return self._get_percent_from_regions(img_frame, loc_size_bars)

        if has_cached_regions:
            return self._get_percent_from_regions(img_frame, self.loc_size_bars)

        return (None, None, None)

    def _monitor_loop(self):
        '''
        Main monitoring loop running in separate thread
        '''
        while not self.is_terminated:
            try:
                if not self.enabled:
                    self.limit_fps()
                    continue

                # Get current time
                t_cur = time.time()

                # Get current HP/MP ratios
                hp_percent, mp_percent, exp_percent = self.get_hp_mp_exp_percent()
                if hp_percent is not None:
                    # Check if HP bar has reduced
                    if self.hp_percent > hp_percent:
                        self.t_last_hp_reduce = t_cur
                    self.hp_percent = hp_percent
                if mp_percent is not None:
                    self.mp_percent = mp_percent
                if exp_percent is not None:
                    self.exp_percent = exp_percent

                hp_thres = self.cfg["health_monitor"]["add_hp_percent"]
                mp_thres = self.cfg["health_monitor"]["add_mp_percent"]
                hp_cd    = self.cfg["health_monitor"]["add_hp_cooldown"]
                mp_cd    = self.cfg["health_monitor"]["add_mp_cooldown"]
                watchdog_timeout = self.cfg["health_monitor"]["return_home_watch_dog_timeout"]

                # Check if need to heal (with cooldown)
                if self.cfg["health_monitor"]["force_heal"]:
                    # Ignore cooldown and force keycontroller to heal first
                    if self.hp_percent < hp_thres:
                        if not self.kb.is_need_force_heal:
                            logger.info(f"[Health Monitor] Force heal triggered, "
                                        f"HP: {self.hp_percent:.1f}%")
                        self.kb.is_need_force_heal = True
                    else:
                        self.kb.is_need_force_heal = False
                else:
                    if (self.hp_percent <= hp_thres and
                        t_cur - self.t_last_heal > hp_cd):
                        self._heal()
                        logger.info(f"[Health Monitor] Auto heal triggered, HP: {self.hp_percent:.1f}%")
                        self.t_last_heal = t_cur

                # Check if no HP potion and need to return home
                if self.cfg["health_monitor"]["return_home_if_no_potion"]:
                    if self.hp_percent >= hp_thres:
                        self.t_hp_watch_dog = t_cur # reset watchdog
                    else:
                        # If watchdog timeout, use homing scroll to return home
                        if t_cur - self.t_hp_watch_dog > watchdog_timeout:
                            logger.warning(f"[Health Monitor] HP({self.hp_percent:.1f}%) < {hp_thres:.1f}% "
                                           f"for {round(t_cur - self.t_hp_watch_dog, 2)} seconds.")
                            logger.warning(f"[Health Monitor] Return home because potion is used up.")
                            press_key(self.cfg["key"]["return_home"]) # Return home
                            self.is_terminated = True # Terminate Health monitor
                            self.kb.is_terminated = True # Terminate AutoBot

                # Check if need MP (with cooldown)
                if (self.mp_percent <= mp_thres and t_cur - self.t_last_mp > mp_cd):
                    self._add_mp()
                    self.t_last_mp = t_cur
                    logger.info(f"[Health Monitor] Auto MP triggered, MP: {self.mp_percent:.1f}%")

                # Sleep to avoid excessive CPU usage
                self.limit_fps()

            except Exception as e:
                logger.error(f"[Health Monitor] {e}")
                self.limit_fps()

    def _heal(self):
        '''
        Execute heal action
        '''
        try:
            press_key(self.cfg["key"]["add_hp"], 0.05)
        except (KeyError, ValueError) as e:
            logger.error(f"[Health Monitor] Invalid heal key configuration: {e}")
        except (OSError, RuntimeError) as e:
            logger.error(f"[Health Monitor] Heal action system error: {e}")
        except Exception as e:
            logger.error(f"[Health Monitor] Unexpected heal action error: {e}")

    def _add_mp(self):
        '''
        Execute MP recovery action
        '''
        try:
            press_key(self.cfg["key"]["add_mp"], 0.05)
        except (KeyError, ValueError) as e:
            logger.error(f"[Health Monitor] Invalid MP key configuration: {e}")
        except (OSError, RuntimeError) as e:
            logger.error(f"[Health Monitor] MP action system error: {e}")
        except Exception as e:
            logger.error(f"[Health Monitor] Unexpected MP action error: {e}")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
