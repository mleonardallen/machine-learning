import importlib
import numpy as np
from pydispatch import dispatcher

class Board(object):

    colors = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'orange': (255, 128, 0)
    }

    def __init__(self, **kwargs):

        self.grid_size = kwargs.get('grid_size')
        self.update_delay = kwargs.get('update_delay', 1)
        self.frame_delay = max(1, int(self.update_delay * 1000)) 

        self.bg_color = self.colors['white']
        self.block_size = 150

        self.pygame = importlib.import_module('pygame')
        self.pygame.init()

        self.font = self.pygame.font.Font(None, 128)

        self.rects = []

        self.size = (
            (self.grid_size[0]) * self.block_size,
            (self.grid_size[1]) * self.block_size
        )
        self.screen = self.pygame.display.set_mode(self.size)


    def render(self, state):

        # Clear screen
        self.screen.fill(self.bg_color)

        #draw lines
        for i in range(self.grid_size[0]):
            self.pygame.draw.line(
                self.screen,
                (0,0,0),
                (i*self.block_size, self.size[0]),
                (i*self.block_size,0)
            )

        for i in range(self.grid_size[1]):
            self.pygame.draw.line(
                self.screen,
                self.colors['black'],
                (self.size[1], i*self.block_size),
                (0, i*self.block_size)
            )

        filled = map(tuple, np.transpose(np.where(state != 0)))
        for i in filled:
            # render text
            char = 'X' if state[i] == 1 else 'O'
            label = self.font.render(char, True, self.colors['orange'])
            self.screen.blit(label, (
                i[1] * self.block_size + self.block_size / 3.5,
                i[0] * self.block_size + self.block_size / 3.5
            ))

        self.mouseClick(state)

        # Flip buffers
        self.pygame.display.flip()

    def mouseClick(self, state):

        mousepos = self.pygame.mouse.get_pos()
        for event in self.pygame.event.get():
            if event.type != self.pygame.MOUSEBUTTONDOWN:
                return

            empty = map(tuple, np.transpose(np.where(state == 0)))
            for i in empty:
                rect = self.pygame.Rect(
                    i[1] * self.block_size,
                    i[0] * self.block_size,
                    self.block_size,
                    self.block_size
                )
                if rect.collidepoint(mousepos):
                    dispatcher.send( signal='board.click', sender=self, action=i)

    def pause(self):
        pass
        # abs_pause_time = time.time()
        # pause_text = "[PAUSED] Press any key to continue..."
        # self.screen.blit(self.font.render(pause_text, True, self.colors['cyan'], self.bg_color), (100, self.height - 40))
        # self.pygame.display.flip()
        # print pause_text  # [debug]
        # while self.paused:
        #     for event in self.pygame.event.get():
        #         if event.type == self.pygame.KEYDOWN:
        #             self.paused = False
        #     self.pygame.time.wait(self.frame_delay)
        # self.screen.blit(self.font.render(pause_text, True, self.bg_color, self.bg_color), (100, self.height - 40))
        # self.start_time += (time.time() - abs_pause_time)