"""
Color Tools MCP Server - FastAPI implementation
Provides endpoints for color tools operations
"""

import sys
from pathlib import Path
from fastapi import FastAPI
import uvicorn
from fastapi_mcp import FastApiMCP

# Ensure project root is on sys.path for model imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Routers and shared state
from routers import colorTools_router

app = FastAPI(
    title="Color Tools MCP Server",
    description="A FastAPI server for color tools operations",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Mount routers (paths unchanged)
app.include_router(colorTools_router)

if __name__ == "__main__":
    mcp = FastApiMCP(app, exclude_operations=[])
    mcp.mount_http()
    uvicorn.run(app, host="0.0.0.0", port=8973)
