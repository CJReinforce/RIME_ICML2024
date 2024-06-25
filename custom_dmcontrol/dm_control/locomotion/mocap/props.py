# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Lint as: python2, python3
"""Props that are constructed from motion-capture data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.mocap import mocap_pb2
import numpy as np

_DEFAULT_LIGHT_PROP_RGBA = np.array([0.77, 0.64, 0.21, 1.])
_DEFAULT_LIGHT_PROP_MASS = 3.

_DEFAULT_HEAVY_PROP_RGBA = np.array([0.77, 0.34, 0.21, 1.])
_DEFAULT_HEAVY_PROP_MASS = 10.

_PROP_SHAPE = {
    mocap_pb2.Prop.SPHERE: 'sphere',
    mocap_pb2.Prop.BOX: 'box',
}


def _default_prop_rgba(prop_mass):
  normalized_mass = np.clip(
      (prop_mass - _DEFAULT_LIGHT_PROP_MASS) /
      (_DEFAULT_HEAVY_PROP_MASS - _DEFAULT_LIGHT_PROP_MASS), 0., 1.)
  return ((1 - normalized_mass) * _DEFAULT_LIGHT_PROP_RGBA +
          normalized_mass * _DEFAULT_HEAVY_PROP_RGBA)


class Prop(composer.Entity):
  """A prop that is constructed from motion-capture data."""

  def _build(self, prop_proto, rgba=None, priority_friction=False):
    rgba = rgba or _default_prop_rgba(prop_proto.mass)
    self._mjcf_root = mjcf.RootElement(model=str(prop_proto.name))
    self._geom = self._mjcf_root.worldbody.add(
        'geom', type=_PROP_SHAPE[prop_proto.shape],
        size=prop_proto.size, mass=prop_proto.mass, rgba=rgba)
    if priority_friction:
      self._geom.priority = 1
      self._geom.condim = 6
      # Torsional and rolling friction have units of length which correspond
      # to the scale of the surface contact "patch" that they approximate.
      self._geom.friction = [.7, prop_proto.size[0]/4, prop_proto.size[0]/2]

    self._body_geom_ids = ()

  @property
  def mjcf_model(self):
    return self._mjcf_root

  def update_with_new_prop(self, prop):
    self._geom.size = prop.geom.size
    self._geom.mass = prop.geom.mass
    self._geom.rgba = prop.geom.rgba

  @property
  def geom(self):
    return self._geom

  def after_compile(self, physics, random_state):
    del random_state  # unused
    self._body_geom_ids = (physics.bind(self._geom).element_id,)

  @property
  def body_geom_ids(self):
    return self._body_geom_ids
