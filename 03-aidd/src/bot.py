#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram бот с ролью Эксперт кино.
Ведет диалог с пользователями и помогает с рекомендациями фильмов и сериалов.
"""
import os
import logging
from dotenv import load_dotenv

class FilmExpertBot:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        pass

