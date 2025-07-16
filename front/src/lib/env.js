import { PUBLIC_API_URL } from '$env/static/public';   // résolue au build

export const API_URL = PUBLIC_API_URL || 'http://localhost:8000/api/v1';