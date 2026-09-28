"""Vercel WSGI entrypoint for the existing Flask dashboard."""

from src.webapp.app import create_app

app = create_app()
